docker run -ti --gpus '"device='$1'"' -v /home/yalew/project/rl/RL_agents:/app --ipc=host --name $2 flip /bin/bash