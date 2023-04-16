import gymnasium as gym
import numpy as np
from collections import deque
import torch
import wandb
import argparse
from buffer import ReplayBuffer
import glob
import random

import yaml
import os
from matplotlib import animation
import matplotlib.pyplot as plt

from utils import *
from networks import *
from bitflip import *
from termcolor import cprint
def render(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ckpt = config["model_ckpt"]
    assert model_ckpt != ""
    model_name = model_ckpt.split(".")[1].split("/")[-2]

    
    env = BitFlipEnv()
    env._make(config["n_bits"])
    n_observations = env.n_observation
    n_actions = env.n_action
    n_goals = env.n_goal

    ddqn = DDQN(obs_dim = n_observations, 
                action_dim = n_observations,
                goal_dim = n_goals,
                config = config) 

    checkpoint = torch.load(model_ckpt, map_location=lambda storage, loc: storage)
    ddqn.q_action.load_state_dict(checkpoint["q_action"])
    print("Model: {} Loaded!".format(model_ckpt))

    with torch.no_grad():
        for x in range(10):
            cprint(f"===========episode_{x} start=============", "red")
            state, goal = env._reset()
            cprint(f"Current State: {np.array2string(state)} / Goal: {np.array2string(goal)}", "green", "on_cyan")
            truncated = False
            terminated = False
            while not (truncated or terminated):
                # 일정 Step 이상 되었을 떄 done이 되지 않으면, gif render가 되지 않는다.
                action = ddqn.get_action(torch.tensor(state).float().unsqueeze(0), torch.tensor(goal).float().unsqueeze(0), eps =0.)
                observation, reward, terminated, truncated = env._step(action)
                env._render()

                

        # env.close()


if __name__ == "__main__":
    config = get_config()
    render(config)
