"""
Utility functions for Dynamic Programming in Gridworld.
Placed in: src/utils/utils.py
"""

import numpy as np

def build_state_index_maps(env):
    state_to_index = {}
    index_to_state = {}

    idx = 0
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            state_to_index[(r, c)] = idx
            index_to_state[idx] = (r, c)
            idx += 1

    env.state_to_index = state_to_index
    env.index_to_state = index_to_state


def print_value_function(V, env):
    for r in range(env.n_rows):
        row_vals = []
        for c in range(env.n_cols):
            cell = env.grid[r][c]
            if cell == "#":
                row_vals.append("  #   ")
            elif cell == "T":
                row_vals.append(" -10  ")
            elif cell == "G":
                row_vals.append("  10  ")
            else:
                idx = env.state_to_index[(r, c)]
                row_vals.append(f"{V[idx]:5.2f}")
        print(" ".join(row_vals))
    print()


def print_policy(policy, env):
    arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    for r in range(env.n_rows):
        row_arrows = []
        for c in range(env.n_cols):
            cell = env.grid[r][c]
            if cell == "#":
                row_arrows.append("#")
            elif cell in ("T", "G"):
                row_arrows.append(cell)
            else:
                idx = env.state_to_index[(r, c)]
                a = np.argmax(policy[idx])
                row_arrows.append(arrows[a])
        print(" ".join(row_arrows))
    print()


def print_policy_grid(policy, env):
    print_policy(policy, env)
