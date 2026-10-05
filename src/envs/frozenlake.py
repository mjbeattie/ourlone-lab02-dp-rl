import gymnasium as gym

def make_env(env_name: str, seed: int = 0):
    custom_map = [
        "SFFF",
        "FFFF",
        "FFHF",
        "FFFG",
    ]

    env = gym.make(env_name, is_slippery=False, desc=custom_map)
    env.reset(seed=seed)
    return env
