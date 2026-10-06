"""
Reinforcement Learning Lab 02: Dynamic Programming
Dynamic Programming algorithms for Gridworld.
Instructor solution
Placed in: src/dp_solution.py
"""

import numpy as np

def policy_evaluation(env, policy, gamma=0.99, theta=1e-6):
    n_states = env.n_rows * env.n_cols
    V = np.zeros(n_states)

    while True:
        delta = 0
        for s in range(n_states):
            r, c = env.index_to_state[s]
            if env.grid[r][c] in ("T", "G"):
                continue

            v = 0
            for a, prob in enumerate(policy[s]):
                next_state, reward = env.get_next_state_and_reward((r, c), a)
                v += prob * (reward + gamma * V[next_state])

            delta = max(delta, abs(v - V[s]))
            V[s] = v

        if delta < theta:
            break

    return V


def policy_improvement(env, V, gamma=0.99):
    n_states = env.n_rows * env.n_cols
    n_actions = env.action_space.n

    policy = np.zeros((n_states, n_actions))

    for s in range(n_states):
        r, c = env.index_to_state[s]
        if env.grid[r][c] in ("T", "G"):
            continue

        q_values = np.zeros(n_actions)
        for a in range(n_actions):
            next_state, reward = env.get_next_state_and_reward((r, c), a)
            q_values[a] = reward + gamma * V[next_state]

        best_a = np.argmax(q_values)
        policy[s][best_a] = 1.0

    return policy


def policy_iteration(env, gamma=0.99, theta=1e-6):
    n_states = env.n_rows * env.n_cols
    n_actions = env.action_space.n

    policy = np.ones((n_states, n_actions)) / n_actions

    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)

        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return V, policy


def value_iteration(env, gamma=0.99, theta=1e-6):
    n_states = env.n_rows * env.n_cols
    n_actions = env.action_space.n

    V = np.zeros(n_states)

    while True:
        delta = 0
        for s in range(n_states):
            r, c = env.index_to_state[s]
            if env.grid[r][c] in ("T", "G"):
                continue

            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                next_state, reward = env.get_next_state_and_reward((r, c), a)
                q_values[a] = reward + gamma * V[next_state]

            v_new = np.max(q_values)
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new

        if delta < theta:
            break

    policy = policy_improvement(env, V, gamma)
    return V, policy
