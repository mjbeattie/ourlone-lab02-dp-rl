from gymnasium.envs.registration import register

register(
    id="Gridworld-4x4-v0",
    entry_point="src.envs.gridworld:GridworldEnv",
)
