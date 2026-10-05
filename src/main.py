import argparse
import numpy as np

from src.envs.make_grid import make_env
from src.agents.random_agent import RandomAgent
from src.utils.plotting import plot_rewards

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="FrozenLake-v1")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

def run_episode(env, agent):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        env.render()
        total_reward += reward
        done = terminated or truncated

    return total_reward

def main():
    args = parse_args()
    env = make_env(args.env, seed=args.seed)
    agent = RandomAgent(env)

    rewards = []
    for ep in range(args.episodes):
        ep_reward = run_episode(env, agent)
        rewards.append(ep_reward)
        print(f"Episode {ep}: reward={ep_reward}")

    print("Average reward:", np.mean(rewards))
    plot_rewards(rewards)

if __name__ == "__main__":
    main()
