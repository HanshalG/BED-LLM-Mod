from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts import revengebench_deterministic_replay_audit as audit


def _arena(status: str) -> dict:
    return {"status": status}


def test_adjudication_passes_four_with_one_infrastructure_exclusion() -> None:
    result = audit.adjudicate(
        {
            "a": _arena("pass"),
            "b": _arena("pass"),
            "c": _arena("pass"),
            "d": _arena("pass"),
            "e": _arena("infrastructure_pending"),
        }
    )

    assert result["status"] == "pass"
    assert result["decision"] == "advance_to_frozen_source_opportunity_audit"
    assert result["summary"] == {
        "required_passes": 4,
        "pass_count": 4,
        "fail_count": 0,
        "infrastructure_pending_count": 1,
    }


def test_adjudication_fails_three_passes() -> None:
    result = audit.adjudicate(
        {
            "a": _arena("pass"),
            "b": _arena("pass"),
            "c": _arena("pass"),
            "d": _arena("infrastructure_pending"),
            "e": _arena("infrastructure_pending"),
        }
    )

    assert result["status"] == "fail"


def test_adjudication_fails_any_unexplained_mismatch() -> None:
    result = audit.adjudicate(
        {
            "a": _arena("pass"),
            "b": _arena("pass"),
            "c": _arena("pass"),
            "d": _arena("pass"),
            "e": _arena("fail"),
        }
    )

    assert result["status"] == "fail"


def test_halite_audit_parses_complete_exact_trace(tmp_path: Path) -> None:
    trace = {
        "num_players": 2,
        "num_frames": 4,
        "frames": [
            [[[1, 10], [0, 0]]],
            [[[1, 11], [0, 0]]],
            [[[1, 12], [0, 0]]],
            [[[1, 13], [0, 0]]],
        ],
        "moves": [[[1, 0]], [[2, 0]], [[3, 0]]],
    }
    arms = []
    for name in ("left", "right"):
        arm = tmp_path / name
        arm.mkdir()
        (arm / "replay.hlt").write_text(json.dumps(trace), encoding="utf-8")
        arms.append(arm)

    result = audit.audit_halite(*arms)

    assert result["status"] == "pass"
    assert result["counts"]["target_nontrivial_actions_per_arm"] == [3, 3]
    assert all(result["gates"].values())


def test_huskybench_audit_parses_target_connection_order(tmp_path: Path) -> None:
    game = {
        "playerNames": {"7": "target", "9": "opponent"},
        "rounds": {
            "1": {
                "action_sequence": [
                    {"player": 7, "action": "raise"},
                    {"player": 9, "action": "call"},
                ]
            }
        },
    }
    arms = []
    for name in ("left", "right"):
        arm = tmp_path / name
        arm.mkdir()
        for index in range(3):
            (arm / f"game_log_{index}_exact.json").write_text(json.dumps(game), encoding="utf-8")
        arms.append(arm)

    result = audit.audit_huskybench(*arms)

    assert result["status"] == "pass"
    assert result["counts"]["target_nontrivial_actions_per_arm"] == [3, 3]
    assert all(result["gates"].values())


def test_robocode_audit_accepts_release_zip_footer(tmp_path: Path) -> None:
    turns = []
    for index in range(4):
        turns.append(
            '<turn><robots><robot id="0" x="%d" y="0" bodyHeading="0" '
            'gunHeading="0" radarHeading="0" energy="100" state="alive" />'
            '<robot id="1" x="9" y="9" /></robots></turn>' % index
        )
    record = ("<record><turns>" + "".join(turns) + "</turns></record>").encode("utf-8")
    # The release appends an empty ZIP archive after the XML record.
    footer_path = tmp_path / "footer.zip"
    with zipfile.ZipFile(footer_path, "w"):
        pass
    raw = record + footer_path.read_bytes()
    paths = []
    results = []
    for name in ("left", "right"):
        record_path = tmp_path / f"{name}.br.xml"
        result_path = tmp_path / f"{name}.results"
        record_path.write_bytes(raw)
        result_path.write_text("target 100\nopponent 50\n", encoding="utf-8")
        paths.append(record_path)
        results.append(result_path)

    result = audit.audit_robocode(paths[0], paths[1], results[0], results[1])

    assert result["status"] == "pass"
    assert result["counts"]["target_nontrivial_actions_per_arm"] == [3, 3]
    assert all(result["gates"].values())
