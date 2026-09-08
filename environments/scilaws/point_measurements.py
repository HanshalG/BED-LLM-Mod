"""Evaluator-side point queries with durable exposure accounting and paired noise.

This is NOT a sandbox. Keep this object, database, key and simulator outside the
policy process/filesystem; expose only request/response JSON through a broker.
No benchmark loader or automatic scientific authorization is provided here.
"""

import hashlib
import hmac
import json
import math
from contextlib import contextmanager
from pathlib import Path
import sqlite3


class MeasurementError(RuntimeError):
    pass


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class PointMeasurements:
    def __init__(
        self,
        simulator,
        *,
        database,
        bounds,
        target,
        budget,
        pairing_key,
        world_id,
        episode_id,
        arm_id,
        runtime_binding,
    ):
        if type(bounds) is not dict or not bounds or len(bounds) > 3:
            raise ValueError("1 to 3 declared input bounds required")
        if any(
            type(k) is not str or not k or k in {"seed", "n_samples", "group_id"}
            for k in bounds
        ):
            raise ValueError("invalid input names")
        if any(
            type(v) not in (list, tuple)
            or len(v) != 2
            or not all(finite(x) for x in v)
            or v[0] >= v[1]
            for v in bounds.values()
        ):
            raise ValueError("finite nondegenerate bounds required")
        if type(target) is not str or not target or target in bounds:
            raise ValueError("distinct target name required")
        if type(budget) is not int or not 1 <= budget <= 10000:
            raise ValueError("measurement budget must be 1..10000")
        if type(pairing_key) is not bytes or len(pairing_key) < 32:
            raise ValueError("private pairing key of at least 32 bytes required")
        if any(
            type(v) is not str or not v
            for v in (world_id, episode_id, arm_id, runtime_binding)
        ):
            raise ValueError("explicit world/episode/arm/runtime bindings required")
        self.simulator = simulator
        self.bounds = {k: tuple(v) for k, v in sorted(bounds.items())}
        self.target = target
        self.budget = budget
        self._key = pairing_key
        self._pair = (world_id, episode_id)
        self._database = str(Path(database))
        self._binding = encoded(
            dict(
                schema=1,
                bounds=self.bounds,
                target=target,
                budget=budget,
                world_id=world_id,
                episode_id=episode_id,
                arm_id=arm_id,
                runtime_binding=runtime_binding,
                key_sha256=hashlib.sha256(pairing_key).hexdigest(),
            )
        )
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS binding (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS attempts (request_id TEXT PRIMARY KEY, round INTEGER UNIQUE NOT NULL, payload TEXT NOT NULL, exposure INTEGER NOT NULL, status TEXT NOT NULL, response TEXT)"
            )
            db.execute("INSERT OR IGNORE INTO binding VALUES (1, ?)", (self._binding,))
            self._check_binding(db)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self._database, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _check_binding(self, db):
        if db.execute("SELECT value FROM binding WHERE id=1").fetchone() != (
            self._binding,
        ):
            raise MeasurementError("measurement ledger binding mismatch")

    def _request(self, request):
        if type(request) is not dict or set(request) != {
            "request_id",
            "point",
            "replicates",
        }:
            raise ValueError("exact request_id, point and replicates fields required")
        rid, point, count = (
            request["request_id"],
            request["point"],
            request["replicates"],
        )
        if type(rid) is not str or not 1 <= len(rid) <= 128:
            raise ValueError("bounded request ID required")
        if type(count) is not int or not 1 <= count <= 32:
            raise ValueError("replicates must be 1..32")
        if type(point) is not dict or set(point) != set(self.bounds):
            raise ValueError("exact declared point coordinates required")
        if any(
            not finite(point[k]) or not lo <= point[k] <= hi
            for k, (lo, hi) in self.bounds.items()
        ):
            raise ValueError(
                "finite in-support coordinates required; clipping forbidden"
            )
        return rid, {k: float(point[k]) for k in self.bounds}, count

    def query(self, request):
        rid, point, count = self._request(request)
        payload = encoded(dict(point=point, replicates=count))
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._check_binding(db)
            previous = db.execute(
                "SELECT payload,status,response FROM attempts WHERE request_id=?",
                (rid,),
            ).fetchone()
            if previous:
                if previous[0] != payload:
                    raise MeasurementError(
                        "request ID cannot be reused for a different measurement"
                    )
                if previous[1] == "complete":
                    return json.loads(previous[2])
                raise MeasurementError(
                    "unresolved or failed attempt cannot be replayed"
                )
            if db.execute(
                "SELECT 1 FROM attempts WHERE status != 'complete'"
            ).fetchone():
                raise MeasurementError(
                    "measurement session halted after unresolved or failed attempt"
                )
            used, round_index = db.execute(
                "SELECT COALESCE(SUM(exposure),0), COUNT(*) FROM attempts"
            ).fetchone()
            if used + count > self.budget:
                raise MeasurementError("measurement budget exhausted")
            db.execute(
                "INSERT INTO attempts VALUES (?,?,?,?,?,NULL)",
                (rid, round_index, payload, count, "pending"),
            )
        # Exposure is durable before the first simulator operation. Partial or
        # uncertain failures consume the complete reserved replicate budget.
        try:
            observations = []
            for replicate in range(count):
                seed = int.from_bytes(
                    hmac.new(
                        self._key,
                        encoded([*self._pair, round_index, replicate]).encode(),
                        hashlib.sha256,
                    ).digest()[:16],
                    "big",
                )
                response = self.simulator.fetch_data(
                    **{k: [v] for k, v in point.items()},
                    n_samples=1,
                    seed=seed,
                )
                if (
                    type(response) is not dict
                    or "error" in response
                    or type(response.get("n_returned")) is not int
                    or response["n_returned"] != 1
                    or type(response.get("n_clipped")) is not int
                    or response["n_clipped"] != 0
                ):
                    raise MeasurementError("invalid measurement response")
                rows = response.get("rows")
                if (
                    type(rows) is not list
                    or len(rows) != 1
                    or type(rows[0]) is not dict
                ):
                    raise MeasurementError("invalid measurement rows")
                row = rows[0]
                if any(
                    not finite(row.get(k)) or row[k] != v for k, v in point.items()
                ) or not finite(row.get(self.target)):
                    raise MeasurementError("invalid realized coordinates or target")
                observations.append(float(row[self.target]))
            result = dict(
                request_id=rid,
                round=round_index,
                point=point,
                observations=observations,
                used=used + count,
                remaining=self.budget - used - count,
            )
            with self._connect() as db:
                db.execute(
                    "UPDATE attempts SET status='complete',response=? WHERE request_id=? AND status='pending'",
                    (encoded(result), rid),
                )
            return result
        except Exception:
            with self._connect() as db:
                db.execute(
                    "UPDATE attempts SET status='failed' WHERE request_id=? AND status='pending'",
                    (rid,),
                )
            # Never reflect simulator source, paths, secrets or raw error text.
            raise MeasurementError(
                "measurement attempt failed; session halted"
            ) from None
