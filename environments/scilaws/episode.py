"""Fixed-budget policy/evaluator exchange, without hidden targets in policy views.

No LLM client, benchmark loader or scientific gate bypass. A study must bind its
trusted evaluator command and stage a public-only policy directory beforehand.
"""

import hashlib
import json
import os
from pathlib import Path

from .point_measurements import encoded, finite
from .policy_isolation import _run_process, run_policy


def strict_json(raw):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError("duplicate JSON key")
            obj[key] = value
        return obj

    def invalid(value):
        raise ValueError("nonfinite JSON constant")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


class BoundedPointBackend:
    """Trusted evaluator subprocess; its raw response never enters policy input.

    Start one worker per point/replicate. Persistent charged exposure belongs to
    PointMeasurements in the broker, not the disposable evaluator process.
    This command may read hidden state and must never originate from the policy.
    """

    def __init__(self, command, *, cwd, timeout=5.0):
        self.command = tuple(command)
        self.cwd = Path(cwd).resolve(strict=True)
        self.timeout = timeout

    def fetch_data(self, **kwargs):
        output = _run_process(
            self.command,
            cwd=self.cwd,
            input_bytes=encoded(kwargs).encode(),
            timeout=self.timeout,
            output_limit=65536,
        )
        return strict_json(output)


def run_episode(
    policy_command,
    *,
    public_root,
    public_task,
    target_points,
    measurements,
    rounds,
    replicates,
    journal_path,
    policy_timeout=5.0,
):
    """Exactly B measured actions followed by one finite prediction vector.

    Journal creation is exclusive; interrupted/failed episodes cannot restart on
    this path. This is not a crash-resume protocol. Endpoint labels are absent.
    The caller controls the public task payload and must use an audited projection.
    """
    if type(rounds) is not int or not 1 <= rounds <= 100:
        raise ValueError("rounds must be 1..100")
    if type(replicates) is not int or not 1 <= replicates <= 32:
        raise ValueError("replicates must be 1..32")
    if rounds * replicates != measurements.budget:
        raise ValueError("episode must match the full declared measurement budget")
    if type(target_points) is not list or not 1 <= len(target_points) <= 1000:
        raise ValueError("fixed nonempty target point list required")
    if any(
        type(p) is not dict
        or set(p) != set(measurements.bounds)
        or not all(finite(v) for v in p.values())
        for p in target_points
    ):
        raise ValueError("finite fixed target coordinates required")
    # Copy the public contract before any process call; reject nonfinite values.
    base = strict_json(
        encoded(
            dict(
                task=public_task,
                target_points=target_points,
                rounds=rounds,
                replicates=replicates,
            )
        )
    )
    journal_path = Path(journal_path)
    root = Path(public_root).resolve(strict=True)
    if journal_path.resolve().is_relative_to(root):
        raise ValueError("evaluator journal must be outside policy directory")
    history = []
    with journal_path.open("x") as journal:

        def record(value):
            journal.write(encoded(value) + "\n")
            journal.flush()
            os.fsync(journal.fileno())

        record(
            dict(
                status="started",
                public_contract_sha256=hashlib.sha256(
                    encoded(base).encode()
                ).hexdigest(),
            )
        )
        try:
            measurements.claim_episode(str(journal_path.resolve()))
            for step in range(rounds + 1):
                phase = "measure" if step < rounds else "predict"
                packet = dict(**base, phase=phase, round=step, history=history)
                output = run_policy(
                    policy_command,
                    public_root=root,
                    input_bytes=encoded(packet).encode(),
                    timeout=policy_timeout,
                )
                action = strict_json(output)
                if phase == "measure":
                    if type(action) is not dict or set(action) != {"point"}:
                        raise ValueError("policy must return exactly point")
                    # IDs and replicate count belong to the broker, not policy.
                    observation = measurements.query(
                        dict(
                            request_id=f"round-{step}",
                            point=action["point"],
                            replicates=replicates,
                        )
                    )
                    if (
                        observation["round"] != step
                        or observation["used"] != (step + 1) * replicates
                    ):
                        raise ValueError("unexpected existing measurement history")
                    history.append(observation)
                    record(dict(status="observed", measurement=observation))
                else:
                    if type(action) is not dict or set(action) != {"predictions"}:
                        raise ValueError("policy must return exactly predictions")
                    predictions = action["predictions"]
                    if (
                        type(predictions) is not list
                        or len(predictions) != len(target_points)
                        or not all(finite(p) for p in predictions)
                    ):
                        raise ValueError("finite fixed-length predictions required")
                    result = dict(
                        status="episode_complete",
                        history=history,
                        predictions=predictions,
                        measurements=rounds * replicates,
                    )
                    record(result)
                    return result
        except BaseException:
            record(dict(status="failed_closed", completed_rounds=len(history)))
            raise
