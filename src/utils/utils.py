"""
Utility functions for Dynamic Programming in Gridworld.
Placed in: src/utils/utils.py
"""

import numpy as np


def enumerate_states(env):
    """
    Return a list of all valid (row, col) states.
    Walls (#) are excluded.
    """
    states = []
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            if env.grid[r][c] != "#":
                states.append((r, c))
    return states


def build_state_index_maps(env):
    """
    Build:
        - state_to_index: (row, col) -> int
        - index_to_state: int -> (row, col)
    """
    states = enumerate_states(env)
    state_to_index = {s: i for i, s in enumerate(states)}
    index_to_state = {i: s for i, s in enumerate(states)}
    return state_to_index, index_to_state


def get_next_state_and_reward(env, state, action):
    """
    Deterministic transition model for Gridworld.
    Returns:
        next_state: (row, col)
        reward: float
        terminated: bool
    """
    r, c = state

    # Proposed movement
    if action == 0:      # Up
        nr, nc = r - 1, c
    elif action == 1:    # Right
        nr, nc = r, c + 1
    elif action == 2:    # Down
        nr, nc = r + 1, c
    else:                # Left
        nr, nc = r, c - 1

    # Boundary check
    if not (0 <= nr < env.n_rows and 0 <= nc < env.n_cols):
        nr, nc = r, c

    # Wall check
    if env.grid[nr][nc] == "#":
        nr, nc = r, c

    cell = env.grid[nr][nc]

    # Terminal states
    if cell == "G":
        return (nr, nc), 10, True
    if cell == "T":
        return (nr, nc), -10, True

    # Normal step
    return (nr, nc), -1, False
