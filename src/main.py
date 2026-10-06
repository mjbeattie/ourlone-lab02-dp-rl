"""
Main driver for Lab 2: Dynamic Programming in Gridworld.
Placed in: src/main.py
"""

from src.dp import (
    policy_evaluation,
    policy_improvement,
    policy_iteration,
    value_iteration,
)

from src.envs.gridworld import GridworldEnv
from src.utils.utils import (
    build_state_index_maps,
    print_value_function,
    print_policy,
    print_policy_grid,
)


def main():
    env = GridworldEnv()
    build_state_index_maps(env)

    print("\n=== POLICY ITERATION ===")
    V_pi, policy_pi = policy_iteration(env)
    print_value_function(V_pi, env)
    print_policy(policy_pi, env)
    print_policy_grid(policy_pi, env)

    print("\n=== VALUE ITERATION ===")
    V_vi, policy_vi = value_iteration(env)
    print_value_function(V_vi, env)
    print_policy(policy_vi, env)
    print_policy_grid(policy_vi, env)


if __name__ == "__main__":
    main()
