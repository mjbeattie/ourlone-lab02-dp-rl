"""
Dynamic Programming algorithms for Gridworld.
Placed in: src/dp.py
"""

import numpy as np
from src.utils.utils import (
    enumerate_states,
    build_state_index_maps,
    get_next_state_and_reward,
)


def policy_evaluation(env, policy, gamma=0.99, theta=1e-6):
    """
    Evaluate a given policy using iterative policy evaluation.

    Args:
        env: GridworldEnv
        policy: np.ndarray of shape (n_states, n_actions)
        gamma: discount factor
        theta: convergence threshold

    Returns:
        V: np.ndarray of shape (n_states,)
    """
    state_to_index, index_to_state = build_state_index_maps(env)
    n_states = len(state_to_index)
    n_actions = env.action_space.n

    V = np.zeros(n_states)

    # STUDENT TODO: implement iterative policy evaluation
    raise NotImplementedError


def policy_improvement(env, V, gamma=0.99):
    """
    Improve a policy given a value function.

    Args:
        env: GridworldEnv
        V: np.ndarray of shape (n_states,)
        gamma: discount factor

    Returns:
        policy: np.ndarray of shape (n_states, n_actions)
    """
    state_to_index, index_to_state = build_state_index_maps(env)
    n_states = len(state_to_index)
    n_actions = env.action_space.n

    policy = np.zeros((n_states, n_actions))

    # STUDENT TODO: implement greedy policy improvement
    raise NotImplementedError


def policy_iteration(env, gamma=0.99, theta=1e-6):
    """
    Run full policy iteration:
        1. Evaluate policy
        2. Improve policy
        until convergence.

    Returns:
        V: np.ndarray
        policy: np.ndarray
    """
    state_to_index, index_to_state = build_state_index_maps(env)
    n_states = len(state_to_index)
    n_actions = env.action_space.n

    # Initialize uniform random policy
    policy = np.ones((n_states, n_actions)) / n_actions

    # STUDENT TODO: implement policy iteration loop
    raise NotImplementedError


def value_iteration(env, gamma=0.99, theta=1e-6):
    """
    Run value iteration using the Bellman optimality update.

    Returns:
        V: np.ndarray
        policy: np.ndarray
    """
    state_to_index, index_to_state = build_state_index_maps(env)
    n_states = len(state_to_index)
    n_actions = env.action_space.n

    V = np.zeros(n_states)

    # STUDENT TODO: implement value iteration
    raise NotImplementedError
