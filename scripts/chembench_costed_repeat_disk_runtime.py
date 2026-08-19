"""Bounded-memory canonical stores for the costed-repeat evaluator.

The stores preserve the exact compact JSON byte representation used by the
legacy in-memory integrity hashes. They are ephemeral runtime state; only the
atomic shard JSON has scientific authority.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Callable

from environments.chembench_mopen.mechanics import proposal_key


_ENCODER = json.JSONEncoder(sort_keys=True, separators=(",", ":"))


class SqliteCanonicalMapping(MutableMapping[str, Any]):
    """A single-process mapping ordered by canonical string keys."""

    def __init__(
        self,
        path: Path,
        *,
        decode: Callable[[bytes], Any] | None = None,
        audit_records: bool = False,
        cache_mib: int = 64,
    ) -> None:
        if cache_mib <= 0:
            raise ValueError("SQLite cache size must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"refusing to reuse runtime database: {self.path}")
        self._decode = decode or (lambda value: json.loads(value))
        self._audit_records = bool(audit_records)
        self._closed = False
        self._connection = sqlite3.connect(str(self.path), timeout=60.0)
        self._connection.execute("PRAGMA journal_mode=OFF")
        self._connection.execute("PRAGMA synchronous=OFF")
        self._connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        self._connection.execute("PRAGMA temp_store=FILE")
        self._connection.execute(f"PRAGMA cache_size={-cache_mib * 1024}")
        self._connection.execute(
            "CREATE TABLE records (key TEXT PRIMARY KEY, value BLOB NOT NULL) WITHOUT ROWID"
        )
        self._count = 0
        self._proposal_count = 0
        self._all_complete = True
        self._truth_particles_proposed = 0

    @staticmethod
    def canonical_bytes(value: Any) -> bytes:
        return _ENCODER.encode(value).encode("utf-8")

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("canonical SQLite mapping is closed")

    def __len__(self) -> int:
        self._require_open()
        return self._count

    def __iter__(self) -> Iterator[str]:
        self._require_open()
        cursor = self._connection.execute("SELECT key FROM records ORDER BY key")
        for (key,) in cursor:
            yield str(key)

    def __getitem__(self, key: str) -> Any:
        self._require_open()
        row = self._connection.execute(
            "SELECT value FROM records WHERE key = ?", (str(key),)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return self._decode(bytes(row[0]))

    def __setitem__(self, key: str, value: Any) -> None:
        self.record_once(key, value)

    def __delitem__(self, key: str) -> None:
        del key
        raise TypeError("canonical runtime records are append-only")

    def record_once(self, key: str, value: Any) -> bool:
        """Insert one record, rejecting nondeterministic duplicate values."""

        self._require_open()
        canonical = self.canonical_bytes(value)
        cursor = self._connection.execute(
            "INSERT OR IGNORE INTO records(key, value) VALUES (?, ?)",
            (str(key), sqlite3.Binary(canonical)),
        )
        if cursor.rowcount == 0:
            existing = self._connection.execute(
                "SELECT value FROM records WHERE key = ?", (str(key),)
            ).fetchone()
            if existing is None or bytes(existing[0]) != canonical:
                raise AssertionError("canonical runtime record is not deterministic")
            return False
        self._count += 1
        if self._audit_records:
            self._accumulate_audit(value)
        return True

    def _accumulate_audit(self, record: Mapping[str, Any]) -> None:
        proposal = record["proposal"]
        if not proposal:
            return
        self._proposal_count += 1
        self._all_complete = self._all_complete and (
            len(proposal) == 3
            and len(record["proposal_structures"]) == 1
            and record["edit"] is not None
        )
        self._truth_particles_proposed += bool(record["truth_particle_proposed"])

    def iter_canonical_items(self) -> Iterator[tuple[str, bytes]]:
        """Yield already-canonical values in exact mapping-key order."""

        self._require_open()
        self._connection.commit()
        cursor = self._connection.execute(
            "SELECT key, value FROM records ORDER BY key"
        )
        for key, value in cursor:
            yield str(key), bytes(value)

    def audit_counters(self) -> dict[str, Any]:
        if not self._audit_records:
            raise TypeError("mapping does not contain transition-audit records")
        return {
            "records": self._count,
            "proposals": self._proposal_count,
            "all_complete_three_particle_edits": (
                self._proposal_count > 0 and self._all_complete
            ),
            "truth_particles_proposed": self._truth_particles_proposed,
        }

    def validate_canonical_values(self) -> bool:
        """Check database integrity and canonical round trips without materializing."""

        self._require_open()
        self._connection.commit()
        row = self._connection.execute("PRAGMA integrity_check").fetchone()
        if row != ("ok",):
            return False
        seen = 0
        for _, canonical in self.iter_canonical_items():
            decoded = self._decode(canonical)
            if self.canonical_bytes(decoded) != canonical:
                return False
            seen += 1
        return seen == self._count

    def close(self, *, delete: bool = True) -> None:
        if self._closed:
            return
        self._connection.commit()
        self._connection.close()
        self._closed = True
        if delete:
            for suffix in ("", "-journal", "-shm", "-wal"):
                candidate = Path(f"{self.path}{suffix}")
                if candidate.exists():
                    candidate.unlink()


def _decode_proposal(value: bytes) -> tuple[int, ...]:
    decoded = json.loads(value)
    if not isinstance(decoded, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in decoded
    ):
        raise ValueError("stored proposal is not an integer sequence")
    return tuple(decoded)


class SqliteProposalCache:
    """ProposalCache-compatible exact disk-backed cache."""

    def __init__(
        self,
        proposer: Any,
        path: Path,
        *,
        source_mode: str | None = None,
        cache_mib: int = 64,
    ) -> None:
        self.proposer = proposer
        self.source_mode = source_mode or proposer.mode
        self._cache = SqliteCanonicalMapping(
            path,
            decode=_decode_proposal,
            cache_mib=cache_mib,
        )
        self.hits = 0
        self.misses = 0

    def get(self, state: Any, action: int, outcome: int, seed: int) -> tuple[int, ...]:
        key = proposal_key(self.source_mode, state, action, outcome, seed)
        try:
            proposal = self._cache[key]
        except KeyError:
            proposal = tuple(self.proposer.propose(state, action, outcome, seed))
            self._cache.record_once(key, proposal)
            self.misses += 1
            return proposal
        self.hits += 1
        return tuple(proposal)

    @property
    def frozen_records(self) -> Mapping[str, tuple[int, ...]]:
        return self._cache

    def close(self, *, delete: bool = True) -> None:
        self._cache.close(delete=delete)


class DiskRuntimeStores:
    """Allocate uniquely named ephemeral stores for one difficulty shard."""

    mode = "sqlite_canonical_v1"

    def __init__(self, root: Path, *, cache_mib: int = 64) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_mib = int(cache_mib)
        self._counter = 0
        self._stores: list[Any] = []

    def _path(self, label: str) -> Path:
        self._counter += 1
        safe = "".join(character if character.isalnum() else "-" for character in label)
        return self.root / f"{self._counter:03d}-{safe}.sqlite3"

    def proposal_cache(self, proposer: Any, *, source_mode: str | None = None) -> SqliteProposalCache:
        cache = SqliteProposalCache(
            proposer,
            self._path("proposal-cache"),
            source_mode=source_mode,
            cache_mib=self.cache_mib,
        )
        self._stores.append(cache)
        return cache

    def transition_audit(self, label: str) -> SqliteCanonicalMapping:
        store = SqliteCanonicalMapping(
            self._path(f"transition-audit-{label}"),
            audit_records=True,
            cache_mib=self.cache_mib,
        )
        self._stores.append(store)
        return store

    def close_store(self, store: Any) -> None:
        close = getattr(store, "close", None)
        if close is not None:
            close()

    def close(self) -> None:
        for store in reversed(self._stores):
            self.close_store(store)
        self._stores.clear()
        try:
            self.root.rmdir()
        except OSError:
            pass


def canonical_mapping_items(records: Mapping[str, Any]) -> Iterator[tuple[str, bytes]]:
    """Stream canonical items from either runtime without changing bytes."""

    iterator = getattr(records, "iter_canonical_items", None)
    if iterator is not None:
        yield from iterator()
        return
    for key in sorted(records):
        yield key, SqliteCanonicalMapping.canonical_bytes(records[key])


def close_runtime_mapping(records: Any) -> None:
    close = getattr(records, "close", None)
    if close is not None:
        close()
