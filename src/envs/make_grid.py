import gymnasium as gym

# IMPORTANT: this import triggers registration
import src.envs.register_gridworld

def make_env(env_name: str, seed: int = 0):
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env
