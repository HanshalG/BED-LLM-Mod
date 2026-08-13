"""Deterministic process bootstrap for frozen RevengeBench mechanics replays."""

from __future__ import annotations

import os
import random
import time
import uuid


_SEED = int(os.environ.get("REVENGEBENCH_REPLAY_SEED", "20260813"))
_EPOCH = float(os.environ.get("REVENGEBENCH_REPLAY_EPOCH", str(_SEED)))
_UUID_RNG = random.Random(_SEED ^ 0xA5A5A5A5)

random.seed(_SEED)


def _uuid4() -> uuid.UUID:
    return uuid.UUID(int=_UUID_RNG.getrandbits(128), version=4)


uuid.uuid4 = _uuid4
time.time = lambda: _EPOCH
