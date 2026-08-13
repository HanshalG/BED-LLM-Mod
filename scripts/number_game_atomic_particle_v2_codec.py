#!/usr/bin/env python3
"""Fresh V2 cohort constants over the frozen atomic-particle mechanics."""

from scripts.number_game_atomic_particle_codec import *  # noqa: F403


INTERFACE_VERSION = "number-game-atomic-particle-depth3-2"
TREE_SEEDS = tuple(range(202608167000, 202608167004))
MODEL_SEED_START = 202608160000
MODEL_REQUESTS = 6400
MODEL_SEEDS = tuple(range(MODEL_SEED_START, MODEL_SEED_START + MODEL_REQUESTS))
