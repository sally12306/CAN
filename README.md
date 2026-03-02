# CAN

This repository contains the codes for our paper titled "CAN: A Conflict-Aware Novelty-Weighted Algorithm for Offline-to-Online Crowd Robot Navigation". 



### The overall framework of our CAN algorithm
<img src="https://raw.githubusercontent.com/sally12306/CAN/main/crowd_nav/figure/CAN.png" 
     alt="Logo" 
     width="80%"/>




### Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```
   
### Overview
This repository is organized in two parts:
* crowd_nav/ folder contains configurations and policies used in the simulator.
* crowd_sim/ folder contains the simulation environment.

### A Conflict-Aware Novelty-Weighted Algorithm
Here are the instructions for the Conflict-Aware Novelty-Weighted Algorithm, including preference data collection, reward model training, and offline reinforcement learning, which should be executed inside the crowd_nav/ folder.
1. Preference data collection
```
python mechanism.py
```
2. Reward model training
```
python train_reward_model.py
```
3. Offline Reinforcement Learning
```
cd offline
python iql.py
```


### Credits
This repository contains the code for the following papers:

- [CORL: Research-oriented Deep Offline Reinforcement Learning Library](https://github.com/tinkoff-ai/CORL).


### Contact
If you have any questions or find any bugs, please feel free to open an issue or pull request.




