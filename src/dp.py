"""
Reinforcement Learning Lab 02: Dynamic Programming
Dynamic Programming algorithms for Gridworld.
Placed in: src/dp.py
"""

import numpy as np

def policy_evaluation(env, policy, gamma=0.99, theta=1e-6):
    """
    TODO: Implement iterative policy evaluation.
    policy: shape (n_states, n_actions)
    Returns V: shape (n_states,)
    """
    # TODO: Write your policy evaluation algorithm here.
    raise NotImplementedError("policy_evaluation is not implemented yet.")


def policy_improvement(env, V, gamma=0.99):
    """
    TODO: Implement greedy policy improvement.
    Returns new_policy: shape (n_states, n_actions)
    """
    # TODO: Write your policy improvement algorithm here.
    raise NotImplementedError("policy_improvement is not implemented yet.")


def policy_iteration(env, gamma=0.99, theta=1e-6):
    """
    TODO: Implement full policy iteration.
    Returns (V, policy)
    """
    # TODO: Write your policy iteration algorithm here.
    raise NotImplementedError("policy_iteration is not implemented yet.")


def value_iteration(env, gamma=0.99, theta=1e-6):
    """
    TODO: Implement value iteration.
    Returns (V, policy)
    """
    # TODO: Write your value iteration algorithm here.
    raise NotImplementedError("value_iteration is not implemented yet.")
