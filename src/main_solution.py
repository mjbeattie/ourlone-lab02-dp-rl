"""
Instructor main driver for Lab 2: Dynamic Programming in Gridworld.
This version prints detailed diagnostics and comparisons for teaching.
Placed in: src/main.py
"""

import numpy as np
from src.envs.gridworld import GridworldEnv
from src.dp import (
    policy_evaluation,
    policy_improvement,
    policy_iteration,
    value_iteration,
)
from src.utils.utils import (
    enumerate_states,
    build_state_index_maps,
)


ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


def print_value_function(V, index_to_state):
    print("\n=== Value Function ===")
    for idx, val in enumerate(V):
        r, c = index_to_state[idx]
        print(f"State {(r, c)}: {val:.4f}")


def print_policy(policy, index_to_state):
    print("\n=== Policy (Greedy Arrows) ===")
    for idx, probs in enumerate(policy):
        r, c = index_to_state[idx]
        a = np.argmax(probs)
        print(f"State {(r, c)}: {ARROWS[a]}")


def print_policy_grid(policy, index_to_state, env):
    """
    Pretty-print the policy in a grid layout matching the environment.
    """
    print("\n=== Policy Grid ===")
    grid = [["#" if env.grid[r][c] == "#" else "." for c in range(env.n_cols)]
            for r in range(env.n_rows)]

    for idx, probs in enumerate(policy):
        r, c = index_to_state[idx]
        if grid[r][c] != "#":
            a = np.argmax(probs)
            grid[r][c] = ARROWS[a]

    for row in grid:
        print(" ".join(row))


def main():
    env = GridworldEnv()
    state_to_index, index_to_state = build_state_index_maps(env)

    print("\n======================================")
    print("      LAB 2 — Dynamic Programming")
    print("======================================")

    # -------------------------------
    # POLICY ITERATION
    # -------------------------------
    print("\n>>> Running Policy Iteration...")
    V_pi, policy_pi = policy_iteration(env)

    print("\n--- Policy Iteration Results ---")
    print_value_function(V_pi, index_to_state)
    print_policy(policy_pi, index_to_state)
    print_policy_grid(policy_pi, index_to_state, env)

    # -------------------------------
    # VALUE ITERATION
    # -------------------------------
    print("\n>>> Running Value Iteration...")
    V_vi, policy_vi = value_iteration(env)

    print("\n--- Value Iteration Results ---")
    print_value_function(V_vi, index_to_state)
    print_policy(policy_vi, index_to_state)
    print_policy_grid(policy_vi, index_to_state, env)

    # -------------------------------
    # COMPARISON
    # -------------------------------
    print("\n======================================")
    print("        COMPARISON SUMMARY")
    print("======================================")

    diff = np.max(np.abs(V_pi - V_vi))
    print(f"\nMax difference between PI and VI value functions: {diff:.6f}")

    same_policy = np.allclose(policy_pi, policy_vi)
    print(f"Policies identical (up to ties): {same_policy}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
