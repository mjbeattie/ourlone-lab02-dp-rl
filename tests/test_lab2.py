"""
Autograder tests for Lab 2: Dynamic Programming in Gridworld.

Expected student files:
    - src/dp.py
    - src/utils/utils.py  (optional helpers)

Expected functions in dp.py:
    - policy_evaluation(env, policy, gamma=0.99, theta=1e-6)
    - policy_improvement(env, V, gamma=0.99)
    - policy_iteration(env, gamma=0.99, theta=1e-6)
    - value_iteration(env, gamma=0.99, theta=1e-6)
"""

import numpy as np

from src.envs.gridworld import GridworldEnv
from src.dp import (
    policy_evaluation,
    policy_improvement,
    policy_iteration,
    value_iteration,
)


def enumerate_states(env: GridworldEnv):
    """
    Enumerate all valid (row, col) states in the grid.
    Walls (#) are excluded.
    """
    states = []
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            if env.grid[r][c] != "#":
                states.append((r, c))
    return states


def state_index_maps(env: GridworldEnv):
    """
    Create mappings:
        (row, col) -> index
        index -> (row, col)
    """
    states = enumerate_states(env)
    idx_map = {s: i for i, s in enumerate(states)}
    rev_map = {i: s for i, s in enumerate(states)}
    return idx_map, rev_map


def test_policy_evaluation_converges_random_policy():
    """
    policy_evaluation should return a finite value vector
    for a uniform random policy.
    """
    env = GridworldEnv()
    idx_map, _ = state_index_maps(env)
    n_states = len(idx_map)
    n_actions = env.action_space.n

    # Uniform random policy
    policy = np.ones((n_states, n_actions)) / n_actions

    V = policy_evaluation(env, policy, gamma=0.99, theta=1e-6)

    assert isinstance(V, np.ndarray), "V must be a numpy array"
    assert V.shape == (n_states,), f"V must have shape ({n_states},)"
    assert np.all(np.isfinite(V)), "All values in V must be finite"


def test_policy_improvement_returns_valid_policy():
    """
    policy_improvement should return a valid stochastic policy:
    - shape (n_states, n_actions)
    - rows sum to 1
    """
    env = GridworldEnv()
    idx_map, _ = state_index_maps(env)
    n_states = len(idx_map)
    n_actions = env.action_space.n

    V = np.zeros(n_states)

    policy = policy_improvement(env, V, gamma=0.99)

    assert isinstance(policy, np.ndarray), "policy must be a numpy array"
    assert policy.shape == (n_states, n_actions), (
        f"policy must have shape ({n_states}, {n_actions})"
    )
    assert np.allclose(policy.sum(axis=1), 1.0), "Each row of policy must sum to 1"


def test_policy_iteration_returns_V_and_policy():
    """
    policy_iteration should return:
    - V: value function
    - policy: corresponding policy
    Both must have correct shapes and finite values.
    """
    env = GridworldEnv()
    idx_map, _ = state_index_maps(env)
    n_states = len(idx_map)
    n_actions = env.action_space.n

    V, policy = policy_iteration(env, gamma=0.99, theta=1e-6)

    assert isinstance(V, np.ndarray), "V must be a numpy array"
    assert isinstance(policy, np.ndarray), "policy must be a numpy array"
    assert V.shape == (n_states,), f"V must have shape ({n_states},)"
    assert policy.shape == (n_states, n_actions), (
        f"policy must have shape ({n_states}, {n_actions})"
    )
    assert np.all(np.isfinite(V)), "All values in V must be finite"
    assert np.allclose(policy.sum(axis=1), 1.0), "Each row of policy must sum to 1"


def test_value_iteration_returns_V_and_policy():
    """
    value_iteration should return:
    - V: value function
    - policy: greedy policy w.r.t V
    Both must have correct shapes and finite values.
    """
    env = GridworldEnv()
    idx_map, _ = state_index_maps(env)
    n_states = len(idx_map)
    n_actions = env.action_space.n

    V, policy = value_iteration(env, gamma=0.99, theta=1e-6)

    assert isinstance(V, np.ndarray), "V must be a numpy array"
    assert isinstance(policy, np.ndarray), "policy must be a numpy array"
    assert V.shape == (n_states,), f"V must have shape ({n_states},)"
    assert policy.shape == (n_states, n_actions), (
        f"policy must have shape ({n_states}, {n_actions})"
    )
    assert np.all(np.isfinite(V)), "All values in V must be finite"
    assert np.allclose(policy.sum(axis=1), 1.0), "Each row of policy must sum to 1"


def test_policy_and_value_iteration_consistency():
    """
    Stronger check:
    The greedy policy from value_iteration should be consistent
    with the policy from policy_iteration (up to ties).
    """
    env = GridworldEnv()
    idx_map, _ = state_index_maps(env)
    n_states = len(idx_map)

    V_pi, policy_pi = policy_iteration(env, gamma=0.99, theta=1e-6)
    V_vi, policy_vi = value_iteration(env, gamma=0.99, theta=1e-6)

    assert V_pi.shape == V_vi.shape == (n_states,)
    assert policy_pi.shape == policy_vi.shape

    # Both policies should be deterministic (one action ~1.0)
    pi_max = policy_pi.max(axis=1)
    vi_max = policy_vi.max(axis=1)

    assert np.allclose(pi_max, 1.0, atol=1e-3), (
        "Policy from policy_iteration should be deterministic."
    )
    assert np.allclose(vi_max, 1.0, atol=1e-3), (
        "Policy from value_iteration should be deterministic."
    )
