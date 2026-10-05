import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridworldEnv(gym.Env):
    """
    Simple deterministic Gridworld environment.
    4x4 grid:
        S . . .
        . # . .
        . . . .
        . . T G

    Legend:
        S = Start
        . = Empty
        # = Wall
        T = Trap (terminal, negative reward)
        G = Goal (terminal, positive reward)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        # Define the grid
        self.grid = [
            ["S", ".", ".", "."],
            [".", "#", ".", "."],
            [".", ".", ".", "."],
            [".", ".", "T", "G"],
        ]

        self.n_rows = 4
        self.n_cols = 4

        # Start position
        self.start_state = (0, 0)
        self.state = self.start_state

        # Action space: 0=Up, 1=Right, 2=Down, 3=Left
        self.action_space = spaces.Discrete(4)

        # Observation space: (row, col)
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.n_rows - 1, self.n_cols - 1]),
            dtype=np.int32,
        )

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_state
        return np.array(self.state, dtype=np.int32), {}

    def step(self, action):
        row, col = self.state

        # Proposed next state
        if action == 0:     # Up
            new_row, new_col = row - 1, col
        elif action == 1:   # Right
            new_row, new_col = row, col + 1
        elif action == 2:   # Down
            new_row, new_col = row + 1, col
        else:               # Left
            new_row, new_col = row, col - 1

        # Check bounds
        if not (0 <= new_row < self.n_rows and 0 <= new_col < self.n_cols):
            new_row, new_col = row, col  # stay in place

        # Check wall
        if self.grid[new_row][new_col] == "#":
            new_row, new_col = row, col  # stay in place

        self.state = (new_row, new_col)
        cell = self.grid[new_row][new_col]

        # Rewards and termination
        if cell == "G":
            reward = 10
            terminated = True
        elif cell == "T":
            reward = -10
            terminated = True
        else:
            reward = -1
            terminated = False

        truncated = False  # no time limit

        return (
            np.array(self.state, dtype=np.int32),
            reward,
            terminated,
            truncated,
            {},
        )

    def render(self):
        grid_copy = [row[:] for row in self.grid]
        r, c = self.state
        grid_copy[r][c] = "A"  # agent

        for row in grid_copy:
            print(" ".join(row))
        print()
