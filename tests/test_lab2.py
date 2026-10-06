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

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.dp import (
    policy_evaluation,
    policy_improvement,
    policy_iteration,
    value_iteration,
)
from src.envs.gridworld import GridworldEnv
from src.utils.utils import build_state_index_maps

def test_policy_evaluation_runs():
    env = GridworldEnv()
    build_state_index_maps(env)

    n_states = env.n_rows * env.n_cols
    n_actions = env.action_space.n

    policy = np.ones((n_states, n_actions)) / n_actions
    V = policy_evaluation(env, policy)

    assert V.shape == (n_states,)


def test_policy_improvement_runs():
    env = GridworldEnv()
    build_state_index_maps(env)

    n_states = env.n_rows * env.n_cols
    V = np.zeros(n_states)

    policy = policy_improvement(env, V)
    assert policy.shape == (n_states, env.action_space.n)


def test_policy_iteration_runs():
    env = GridworldEnv()
    build_state_index_maps(env)

    V, policy = policy_iteration(env)
    assert V.shape == (env.n_rows * env.n_cols,)
    assert policy.shape == (env.n_rows * env.n_cols, env.action_space.n)


def test_value_iteration_runs():
    env = GridworldEnv()
    build_state_index_maps(env)

    V, policy = value_iteration(env)
    assert V.shape == (env.n_rows * env.n_cols,)
    assert policy.shape == (env.n_rows * env.n_cols, env.action_space.n)
