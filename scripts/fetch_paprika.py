#!/usr/bin/env python3
"""Fetch the pinned Paprika release and verify customer-service data provenance."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.paprika_customer_service.data import (
    PAPRIKA_COMMIT,
    PAPRIKA_CUSTOMER_SERVICE_RELATIVE_PATH,
    PAPRIKA_CUSTOMER_SERVICE_SHA256,
    PAPRIKA_REPOSITORY,
)


def fetch(destination: Path) -> Path:
    if not destination.exists():
        subprocess.run(["git", "clone", PAPRIKA_REPOSITORY, str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "fetch", "origin", PAPRIKA_COMMIT], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", PAPRIKA_COMMIT], check=True)
    path = destination / PAPRIKA_CUSTOMER_SERVICE_RELATIVE_PATH
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PAPRIKA_CUSTOMER_SERVICE_SHA256:
        raise RuntimeError(f"Paprika data hash mismatch: expected {PAPRIKA_CUSTOMER_SERVICE_SHA256}, got {digest}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("external/paprika"))
    args = parser.parse_args()
    print(fetch(args.destination).resolve())


if __name__ == "__main__":
    main()
