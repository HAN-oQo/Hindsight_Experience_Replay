# import gymnasium as gym 
import numpy as np
import collections, random
from datetime import datetime
import os
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from hindsight_buffer import HindSightReplayBuffer
from bitflip import BitFlipEnv
from networks import *
from utils import *

#Hyperparameters
# lr_pi           = 0.001 #05
# lr_q            = 0.001
# init_alpha      = 0.01 #0.01
# gamma           = 0.98 #0.99
# batch_size      = 32 #256 #32 #256 #32
# buffer_limit    = 50000 #100000
# tau             = 0.005 # for target network soft update
# target_entropy  = -1.0 # for automated alpha update
# lr_alpha        = 0.005 #0.001#0.001 #0.001  # for automated alpha update
    
def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the current date and time
    now = datetime.now()

    # Format the date and time as an integer in the format of 20230753
    formatted = now.strftime("%Y%m%d%H%M%S")
    model_save_dir = os.path.join(config["model_basedir"], config["env"], config["run_name"], formatted)
    os.makedirs(model_save_dir, exist_ok = True)

    # if config["env"] == "Pendulum-v1":
    #     action_range = [-2, 2]
    #     action_scale = 2.
    # else:
    #     action_scale = 1.
    env = BitFlipEnv()
    env._make(config["n_bits"])

    n_observations = env.n_observation
    n_actions = env.n_action
    n_goals = env.n_goal
    
    memory = HindSightReplayBuffer(max_size = config["buffer_size"], 
                                    input_shape= n_observations,
                                    n_actions=n_actions)

    ddqn = DDQN(obs_dim = n_observations, 
                action_dim = n_observations,
                goal_dim = n_goals,
                config = config) 
    ddqn = ddqn.to(device)

    score_history = []
    update_steps = 0
    success = 0
    best_score = -(env.n_goal+1)
    loss, average100 = 0., 0.
    if config["with_her"]:
        prj_name = "HER_{}".format(config["env"])
    else:
        prj_name = "without_HER_{}".format(config["env"])
    with wandb.init(project=prj_name, name="{}_{}".format(now, config["run_name"]), config=config):
        for n_epi in range(config["n_episodes"]):
            s, g = env._reset()
            done = False
            trunc = False
            score = 0.0
            # print(f"episode{n_epi} start")
            eps = config["eps_low"]+(config["eps_high"]-config["eps_low"]) * (np.exp(-1.0 * update_steps/config["eps_decay"]))
            update_steps += 1
            transitions = []
            while not (done or trunc):
                # print("step ++")
                # breakpoint()
                a = ddqn.get_action(torch.tensor(s).float().to(device).unsqueeze(0),
                                torch.tensor(g).float().to(device).unsqueeze(0),
                                eps = eps)
                # a = torch.tensor(a).unsqueeze(0)
                # breakpoint()
                s_prime, r, done, trunc = env._step(a)
                memory.put(s, a, r, s_prime, done, g) #r/10.0
                transitions.append([s, a, r, s_prime, done, g])
                score +=r
                s = s_prime
                if done:
                    success += 1

            if config["with_her"]:
                if not done:
                    g_prime = np.copy(s_prime)
                    for transition in transitions:
                        s, a, r, s_prime, done, g = transition
                        if np.array_equal(s_prime, g_prime):
                            memory.put(s,a,r,s_prime, True, g_prime)
                        else:
                            memory.put(s,a,r,s_prime, False, g_prime)
                    
            if memory.size() > config["start_size"]:
                mini_batch = memory.sample(config["batch_size"])
                mini_batch = HindSightReplayBuffer.batch_to_device(mini_batch, device)
                loss = ddqn.train_net(mini_batch)

            score_history.append(score)

            if n_epi > 100:
                average100 = np.mean(score_history[-100:])

            if n_epi > 100:
                if  average100 > best_score:
                    best_score = average100
                    torch.save({
                        "q_action": ddqn.q_action.state_dict(),
                        "q_eval": ddqn.q_eval.state_dict(),
                        "optim": ddqn.optimizer.state_dict(),
                    }, os.path.join(model_save_dir, "best_score.ckpt"))
                    wandb.save(os.path.join(model_save_dir, "best_score.ckpt"))
                    
            if n_epi%config["log_every"]==0 and n_epi > 0:
                print("# of episode :{}, score1: {:.1f}, score100 : {:.1f}, success_rate: {}, buffer_size: {}".format(n_epi, score, average100, success/config["log_every"], memory.size()))
                wandb.log({"Score_1": score,
                        "Score_100": average100,
                        "Success rate": success/config["log_every"],
                        "Loss":loss,
                        "Update Steps": update_steps,
                        "Episode": n_epi ,
                        "Buffer size": memory.size()})
                score = 0.0
                success = 0

            
            if n_epi%config["save_every"]==0:
                torch.save({
                        "q_action": ddqn.q_action.state_dict(),
                        "q_eval": ddqn.q_eval.state_dict(),
                        "optim": ddqn.optimizer.state_dict(),
                    }, os.path.join(model_save_dir, f"{n_epi}.ckpt"))
                wandb.save(os.path.join(model_save_dir, f"{n_epi}.ckpt"))

        # env.close()

if __name__ == '__main__':
    config = get_config()
    print(config)
    main(config)
