# Hindsight Experience Replay

This repository is the Pytorch implementation of [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)

---

## Pseudocode of the HER algorithm
![Pseudocode](./imgs/her_pseudo.png)

---

## Set up

Install dependencies with Docker. You can also install dependencies using requirements.txt.

To build Docker image, run this command.
```
# format: docker build -t . <image_name>
docker build -t . her
```

After building image, use the following command to run the Docker container.
```
docker run -ti --gpus '"device='<gpu number>'"' -v <your working directory>:/app --ipc=host --name <container_name> <image_name> /bin/bash

# or you can run this command after changing docker_run.sh file in proper format
./docker_run.sh <gpu num> <container_name>
```
---

## Train
If you want to train the agent with HER algorithm in BitFlip environment, run this command inside the Docker container.
```
python train_bitflip.py --config config_bitflip.yaml
```

You can freely change the hyperparameter if you needed.
I tested n_bits in [10, 25, 45] with same hyperparameters except n_bits / max_episode_steps / hidden_units. 
The paper mentioned that nbits < 50 can be trained.

---
## Test

You can test with the pretrained networks.
All pretrained networks for 10bits/25bits/45bits can be downloaded in this [link](https://drive.google.com/drive/folders/1mwkvZ3ijwuyhkWacfXuGledEIZFxO0BM?usp=share_link)

To see the bitflip simulation result with the network, run
```
# before run this command, you should put the path to checkpoint in config file.
python render_bitflip.py --config config_bitflip.yaml
```

---
## Results

The bitflip simulation results with 10 bits.
![simulation_result](./imgs/Results.png)

The bitflip simulation results with 25 bits.
![simulation_result](./imgs/25bit_result.png)

The bitflip simulation results with 45 bits.
![simulation_result](./imgs/45bit_result.png)



The training logs of DDQN + HER with 10bits.
![Training_logs](./imgs/HER_logs.png)

The training logs of DDQN + HER with 25bits.
![Training_logs](./imgs/25bit_log.png)

The training logs of DDQN + HER with 45bits.
![Training_logs](./imgs/45bit_log.png)

The training logs of vanilla DDQN with 10bits.
![Training_logs_vanilla](./imgs/without_her_logs.png)

---
## Acknowlegement
1. https://github.com/hemilpanchiwala/Hindsight-Experience-Replay/tree/b79402af027cbd4321e4ae93aeefccdfd1b7d675
2. https://github.com/orrivlin/Hindsight-Experience-Replay---Bit-Flipping/tree/8f7920ee4ed5236f26549f2d62a760a537e6e6b4
3. https://towardsdatascience.com/hindsight-experience-replay-her-implementation-92eebab6f653
