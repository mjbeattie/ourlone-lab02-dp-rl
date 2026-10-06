"""
A simple gridworld environment for reinforcement learning.
The gridworld is a 4x4 grid for dynamic programming exercises.
Placed in: src/envs/gridworld.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridworldEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        self.grid = [
            ["S", ".", ".", "."],
            [".", "#", ".", "."],
            [".", ".", ".", "."],
            [".", ".", "T", "G"],
        ]

        self.n_rows = 4
        self.n_cols = 4
        self.start_state = (0, 0)
        self.state = self.start_state

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([3, 3]),
            dtype=np.int32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_state
        return np.array(self.state), {}

    def step(self, action):
        r, c = self.state

        if action == 0:
            nr, nc = r - 1, c
        elif action == 1:
            nr, nc = r, c + 1
        elif action == 2:
            nr, nc = r + 1, c
        else:
            nr, nc = r, c - 1

        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
            nr, nc = r, c

        if self.grid[nr][nc] == "#":
            nr, nc = r, c

        self.state = (nr, nc)
        cell = self.grid[nr][nc]

        if cell == "G":
            return np.array(self.state), 10, True, False, {}
        elif cell == "T":
            return np.array(self.state), -10, True, False, {}
        else:
            return np.array(self.state), -1, False, False, {}

    def get_next_state_and_reward(self, state, action):
        r, c = state

        if action == 0:
            nr, nc = r - 1, c
        elif action == 1:
            nr, nc = r, c + 1
        elif action == 2:
            nr, nc = r + 1, c
        else:
            nr, nc = r, c - 1

        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
            nr, nc = r, c

        if self.grid[nr][nc] == "#":
            nr, nc = r, c

        cell = self.grid[nr][nc]
        next_state = self.state_to_index[(nr, nc)]

        if cell == "G":
            return next_state, 10
        elif cell == "T":
            return next_state, -10
        else:
            return next_state, -1

    def render(self):
        grid_copy = [row[:] for row in self.grid]
        r, c = self.state
        grid_copy[r][c] = "A"
        for row in grid_copy:
            print(" ".join(row))
        print()
