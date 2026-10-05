"""
Dynamic Programming algorithms for Gridworld.
Instructor solution key.
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
    """
    state_to_index, index_to_state = build_state_index_maps(env)
    n_states = len(state_to_index)
    n_actions = env.action_space.n

    V = np.zeros(n_states)

    while True:
        delta = 0
        new_V = np.copy(V)

        for idx in range(n_states):
            state = index_to_state[idx]

            v = 0
            for a in range(n_actions):
                prob = policy[idx, a]
                next_state, reward, terminated = get_next_state_and_reward(env, state, a)
                next_idx = state_to_index[next_state]

                if terminated:
                    v += prob * reward
                else:
                    v += prob * (reward + gamma * V[next_idx])

            new_V[idx] = v
            delta = max(delta, abs(V[idx] - new_V[idx]))

        V = new_V

        if delta < theta:
            break

    return V


def policy_improvement(env, V, gamma=0.99):
    """
    Improve a policy given a value function.
    Returns a greedy policy.
    """
    state_to_index, index_to_state = build_state_index_maps(env)
    n_states = len(state_to_index)
    n_actions = env.action_space.n

    policy = np.zeros((n_states, n_actions))

    for idx in range(n_states):
        state = index_to_state[idx]

        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            next_state, reward, terminated = get_next_state_and_reward(env, state, a)
            next_idx = state_to_index[next_state]

            if terminated:
                action_values[a] = reward
            else:
                action_values[a] = reward + gamma * V[next_idx]

        # Greedy action
        best_action = np.argmax(action_values)
        policy[idx, best_action] = 1.0

    return policy


def policy_iteration(env, gamma=0.99, theta=1e-6):
    """
    Full policy iteration:
        1. Evaluate policy
        2. Improve policy
        until stable.
    """
    state_to_index, index_to_state = build_state_index_maps(env)
    n_states = len(state_to_index)
    n_actions = env.action_space.n

    # Start with uniform random policy
    policy = np.ones((n_states, n_actions)) / n_actions

    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)

        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return V, policy


def value_iteration(env, gamma=0.99, theta=1e-6):
    """
    Value iteration using Bellman optimality updates.
    """
    state_to_index, index_to_state = build_state_index_maps(env)
    n_states = len(state_to_index)
    n_actions = env.action_space.n

    V = np.zeros(n_states)

    while True:
        delta = 0
        new_V = np.copy(V)

        for idx in range(n_states):
            state = index_to_state[idx]

            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                next_state, reward, terminated = get_next_state_and_reward(env, state, a)
                next_idx = state_to_index[next_state]

                if terminated:
                    action_values[a] = reward
                else:
                    action_values[a] = reward + gamma * V[next_idx]

            new_V[idx] = np.max(action_values)
            delta = max(delta, abs(V[idx] - new_V[idx]))

        V = new_V

        if delta < theta:
            break

    # Extract greedy policy
    policy = policy_improvement(env, V, gamma)

    return V, policy
