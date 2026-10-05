# Foundations of RL — Lab 01: Introduction to RL and MDPs

## Overview
This lab introduces the basic reinforcement learning loop:
- Agent
- Environment
- States, actions, rewards
- Episodes and returns

You will run a simple environment (FrozenLake or Gridworld), inspect transitions, and produce a short report describing what you observed.

## Setup

#### Step 1 - Create the conda environment by typing these lines into the command line:

conda env create -f environment.yml
conda activate ourlone-lab

#### Step 2 - Run the lab. After you have implemented the code files in src/, run the lab using:

python src/main.py --env FrozenLake-v1 --episodes 20 --seed 0

#### Step 3 - Save a plot of episode rewards to:

results/plots/rewards.png

