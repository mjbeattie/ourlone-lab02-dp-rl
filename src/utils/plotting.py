import os
import matplotlib.pyplot as plt

def plot_rewards(rewards, out_path="results/plots/rewards.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Rewards")
    plt.savefig(out_path)
    plt.close()
