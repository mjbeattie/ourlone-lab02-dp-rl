"""
Main driver for Lab 2: Dynamic Programming in Gridworld.
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


def print_value_function(V, index_to_state):
    print("\nValue Function:")
    for idx, val in enumerate(V):
        r, c = index_to_state[idx]
        print(f"State {(r, c)}: {val:.3f}")


def print_policy(policy, index_to_state):
    arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    print("\nPolicy:")
    for idx, probs in enumerate(policy):
        r, c = index_to_state[idx]
        a = np.argmax(probs)
        print(f"State {(r, c)}: {arrows[a]}")


def main():
    env = GridworldEnv()
    state_to_index, index_to_state = build_state_index_maps(env)

    print("\n=== Policy Iteration ===")
    V_pi, policy_pi = policy_iteration(env)
    print_value_function(V_pi, index_to_state)
    print_policy(policy_pi, index_to_state)

    print("\n=== Value Iteration ===")
    V_vi, policy_vi = value_iteration(env)
    print_value_function(V_vi, index_to_state)
    print_policy(policy_vi, index_to_state)


if __name__ == "__main__":
    main()
