# source: https://github.com/nakamotoo/Cal-QL/tree/main
# https://arxiv.org/pdf/2303.05479.pdf
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
dirPath = os.path.dirname(os.path.realpath(__file__))
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import h5py
# import d4rl
import matplotlib.pyplot as plt
import torchvision
from crowd_sim.envs.utils.info import ReachGoal, Collision, Timeout
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.crowd_sim import CrowdSim

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal, TanhTransform, TransformedDistribution
from crowd_nav.self_attention import SelfAttention
from crowd_nav.policy.st import ST

import argparse
import configparser
import logging
import time
from datetime import datetime, timedelta

TensorBatch = List[torch.Tensor]


# ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "CrowdSim-v0"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 0  # Eval environment seed
    eval_freq: int = int(5000)  # How often (time steps) we evaluate
    n_episodes: int = 500  # How many episodes run during evaluation
    offline_iterations: int = int(100000)  # Number of offline updates
    online_iterations: int = int(50000)  # Number of online updates
    checkpoints_path: Optional[str] = "D:\Program Files (x86)\IDE\JetBrains\PycharmProjects\CrowdNav-master_danren\crowd_nav\data\output_calql"  # Save path
    # load_model: str = "D:\Program Files (x86)\IDE\JetBrains\PycharmProjects\CrowdNav-master_danren\crowd_nav\data\output_calql\Cal-QL-Model\checkpoint_99999.pt"  # Model load file name, "" doesn't load
    load_model: str = "D:\Program Files (x86)\IDE\JetBrains\PycharmProjects\CrowdNav-master_danren\crowd_nav\data\output_calql\Cal-QL-train_log\calql_online_final.pt"
    # /hd_2t/fuhao2024/Yzc/cql_model/rlhf_cql_1764588560.pt
    # CQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    alpha_multiplier: float = 1.0  # Multiplier for alpha in loss
    use_automatic_entropy_tuning: bool = True  # Tune entropy
    # backup_entropy: bool = False  # Use backup entropy
    backup_entropy: bool = True  # ✅ online 阶段更稳
    # policy_lr: float = 3e-5  # Policy learning rate
    policy_lr: float = 1e-4  # ✅ actor 提速
    qf_lr: float = 3e-4  # Critics learning rate
    soft_target_update_rate: float = 5e-3  # Target network update rate
    bc_steps: int = int(0)  # Number of BC steps at start
    target_update_period: int = 1  # Frequency of target nets updates
    # cql_alpha: float = 10.0  # CQL offline regularization parameter
    cql_alpha: float = 1.0  # ❗ 原来 10，太大
    cql_alpha_online: float = 10.0  # CQL online regularization parameter
    cql_n_actions: int = 10  # Number of sampled actions
    cql_importance_sample: bool = True  # Use importance sampling
    cql_lagrange: bool = False  # Use Lagrange version of CQL
    cql_target_action_gap: float = -1.0  # Action gap
    # cql_temp: float = 1.0  # CQL temperature
    cql_temp: float = 2.0
    cql_max_target_backup: bool = False  # Use max target backup
    cql_clip_diff_min: float = -np.inf  # Q-function lower loss clipping
    cql_clip_diff_max: float = np.inf  # Q-function upper loss clipping
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    q_n_hidden_layers: int = 2  # Number of hidden layers in Q networks
    reward_scale: float = 1.0  # Reward scale for normalization
    reward_bias: float = 0.0  # Reward bias for normalization
    # Cal-QL
    # mixing_ratio: float = 0.5  # Data mixing ratio for online tuning
    mixing_ratio: float = 0.5
    is_sparse_reward: bool = False  # Use sparse reward

    policy_log_std_multiplier: float = 1.0

    # meta learning
    latent_dim: int = 8
    context_min_size: int = 32  # latent 推断需要的最小在线 buffer
    # Wandb logging
    project: str = "CORL"
    group: str = "Cal-QL-D4RL"
    name: str = "Cal-QL"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std





class ReplayBuffer:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            buffer_size: int,
            device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._mc_returns = torch.zeros(
            (buffer_size, 1), dtype=torch.float32, device=device
        )

        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._mc_returns[:n_transitions] = self._to_tensor(data["mc_returns"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        mc_returns = self._mc_returns[indices]
        return [states, actions, rewards, next_states, dones, mc_returns]

    def add_transition(
            self, state, action, reward, next_state, done
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)
        self._mc_returns[self._pointer] = 0.0

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)





def set_seed(
        seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def is_goal_reached(reward: float, info: Dict) -> bool:
    if "goal_achieved" in info:
        return info["goal_achieved"]
    return reward > 0  # Assuming that reaching target is a positive reward


cnt = 20000


@torch.no_grad()
def eval_actor(
        env: gym.Env, actor: nn.Module, list1: List, list2: List, list3: List, list4: List, device: str,
        n_episodes: int, seed: int
):
    global cnt
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str,
                        default='D:\Program Files (x86)\IDE\JetBrains\PycharmProjects\CrowdNav-master_danren\crowd_nav\configs\env.config')
    args = parser.parse_args()
    actor.eval()
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    robot.set_policy(actor)
    # robot.print_info()
    phase = 'train'
    success = 0
    collision = 0
    run_time = 0
    avg_nav_time = 0
    avg_reward = 0

    for _ in range(n_episodes):
        observation = env.reset(phase=phase, test_case=cnt)
        done = False
        rewards = 0
        gamma = 1  # 折扣因子
        while not done:
            joint_state = JointState(robot.get_full_state(), observation)
            state = to_np(transform(joint_state, device).view(1, -1).squeeze(0))
            # print(state.shape)  # 输出为（65，）
            action = actor.act(state)
            # print(action.shape)  # 输出为（2，）
            action = ActionXY(action[0], action[1])
            observation, reward, done, info = env.step(action)
            # print(reward)
            rewards += reward * gamma
            gamma = gamma * 0.99
        if isinstance(info, ReachGoal):
            success += 1
            avg_nav_time += env.global_time
        elif isinstance(info, Collision):
            collision += 1
        elif isinstance(info, Timeout):
            run_time += 1

        avg_reward += rewards
        cnt += 1
    if success != 0:
        avg_nav_time = avg_nav_time / success
    else:
        avg_nav_time = 0
    avg_reward = avg_reward / n_episodes
    # print(f"成功次数为{success} 成功率为{success / n_episodes} 碰撞次数{collision} 超时次数{run_time} 平均导航时间为{avg_nav_time} 平均奖励为{avg_reward} ")
    list1.append(success / n_episodes)
    list2.append(collision / n_episodes)
    list3.append(avg_nav_time)
    list4.append(avg_reward)
    actor.train()


@torch.no_grad()
def test_actor(
        env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
):
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str,
                        default='D:\Program Files (x86)\IDE\JetBrains\PycharmProjects\CrowdNav-master_danren\crowd_nav\configs\env.config')
    args = parser.parse_args()

    actor.eval()
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    robot.set_policy(actor)
    # robot.print_info()
    phase = 'test'
    success = 0
    collision = 0
    run_time = 0
    avg_nav_time = 0
    avg_reward = 0
    for _ in range(n_episodes):
        observation, done = env.reset(phase=phase), False
        rewards = 0
        gamma = 1  # 折扣因子
        while not done:
            joint_state = JointState(robot.get_full_state(), observation)
            state = to_np(transform(joint_state, device).view(1, -1).squeeze(0))
            # print(state.shape)  # 输出为（65，）
            action = actor.act(state)
            # print(action.shape)  # 输出为（2，）
            action = ActionXY(action[0], action[1])
            observation, reward, done, info = env.step(action)
            # print(reward)
            rewards += reward * gamma
            gamma = gamma * 0.99
        if isinstance(info, ReachGoal):
            success += 1
            avg_nav_time += env.global_time
        elif isinstance(info, Collision):
            collision += 1
        elif isinstance(info, Timeout):
            run_time += 1
        # print(env.global_time)
        avg_reward += rewards
        env.render(mode='traj')
        plt.show()
    avg_nav_time = avg_nav_time / n_episodes
    avg_reward = avg_reward / n_episodes
    print(
        f"成功次数为{success} 成功率为{success / n_episodes} 碰撞次数{collision} 超时次数{run_time} 平均导航时间为{avg_nav_time} 平均奖励为{avg_reward} ")


def transform(state, device):
    state_tensor = torch.cat([torch.Tensor([state.self_state + human_state]).to(device)
                              for human_state in state.human_states], dim=0)

    state_tensor = rotate(state_tensor)
    return state_tensor


def rotate(state):
    # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
    #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
    batch = state.shape[0]
    dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
    dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
    rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

    dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
    v_pref = state[:, 7].reshape((batch, -1))
    vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
    vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

    radius = state[:, 4].reshape((batch, -1))
    theta = torch.zeros_like(v_pref)
    vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
    vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
    px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
    px1 = px1.reshape((batch, -1))
    py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
    py1 = py1.reshape((batch, -1))
    radius1 = state[:, 13].reshape((batch, -1))
    radius_sum = radius + radius1
    da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                              reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
    # dg = ||p − pg||2 is the robot’s distance to the goal and di = ||p − pi||2 is the robot’s distance to the neighbor i
    new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
    return new_state


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def get_return_to_go(dataset: Dict, env: gym.Env, config: TrainConfig) -> np.ndarray:
    returns = []
    ep_ret, ep_len = 0.0, 0
    cur_rewards = []
    terminals = []
    N = len(dataset["rewards"])
    for t, (r, d) in enumerate(zip(dataset["rewards"], dataset["terminals"])):
        ep_ret += float(r)
        cur_rewards.append(float(r))
        terminals.append(float(d))
        ep_len += 1
        is_last_step = (
                (t == N - 1)
                or (
                        np.linalg.norm(
                            dataset["observations"][t + 1] - dataset["next_observations"][t]
                        )
                        > 1e-6
                )
                or ep_len == 200
        )

        if d or is_last_step:
            discounted_returns = [0] * ep_len
            prev_return = 0
            if (
                    config.is_sparse_reward
                    and r
                    == env.ref_min_score * config.reward_scale + config.reward_bias
            ):
                discounted_returns = [r / (1 - config.discount)] * ep_len
            else:
                for i in reversed(range(ep_len)):
                    discounted_returns[i] = cur_rewards[
                                                i
                                            ] + config.discount * prev_return * (1 - terminals[i])
                    prev_return = discounted_returns[i]
            returns += discounted_returns
            ep_ret, ep_len = 0.0, 0
            cur_rewards = []
            terminals = []
    return returns


def modify_reward(
        dataset: Dict,
        env_name: str,
        max_episode_steps: int = 1000,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
) -> Dict:
    modification_data = {}
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
        modification_data = {
            "max_ret": max_ret,
            "min_ret": min_ret,
            "max_episode_steps": max_episode_steps,
        }
    dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias
    return modification_data


def modify_reward_online(
        reward: float,
        env_name: str,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        **kwargs,
) -> float:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        reward /= kwargs["max_ret"] - kwargs["min_ret"]
        reward *= kwargs["max_episode_steps"]
    reward = reward * reward_scale + reward_bias
    return reward


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Sequential, orthogonal_init: bool = False):
    # Specific orthgonal initialization for inner layers
    # If orthogonal init is off, we do not change default initialization
    if orthogonal_init:
        for submodule in module[:-1]:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)

    # Lasy layers should be initialzied differently as well
    if orthogonal_init:
        nn.init.orthogonal_(module[-1].weight, gain=1e-2)
    else:
        nn.init.xavier_uniform_(module[-1].weight, gain=1e-2)

    nn.init.constant_(module[-1].bias, 0.0)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
            self,
            log_std_min: float = -20.0,
            log_std_max: float = 2.0,
            no_tanh: bool = False,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
            self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
            self,
            mean: torch.Tensor,
            log_std: torch.Tensor,
            deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


class TanhGaussianPolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            max_action: float,
            log_std_multiplier: float = 1.0,
            log_std_offset: float = -1.0,
            orthogonal_init: bool = False,
            no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        # print(f"action_dim:{action_dim}")
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh
        self.kinematics = 'holonomic'
        self.multiagent_training = True

        self.base_network = nn.Sequential(
            # print(f"state:{state_dim}"),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )

        init_module_weights(self.base_network)

        self.attention = ST(13)
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
            self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        observations = observations.reshape(observations.shape[0], -1, 13)
        x = self.attention(observations)
        base_network_output = self.base_network(x)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(
            self,
            observations: torch.Tensor,
            deterministic: bool = False,
            repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
            observations = observations.reshape(observations.shape[0] * observations.shape[1], observations.shape[2])
        observations = observations.reshape(observations.shape[0], -1, 13)
        # 上面2行是ST的代码，换用注意力可以去掉
        # print(observations.shape)
        x = self.attention(observations)
        # print(x.shape)
        if repeat is not None:
            x = x.reshape(256, 10, -1)
        base_network_output = self.base_network(x)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cuda"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class FullyConnectedQFunction(nn.Module):
    def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            orthogonal_init: bool = False,
            n_hidden_layers: int = 3,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init
        # print(f"action_dim:{action_dim}")
        # print(f"observation_dim:{observation_dim}")
        layers = [
            nn.Linear(130, 256),
            nn.ReLU(),
        ]
        for _ in range(2):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))

        self.network = nn.Sequential(*layers)
        self.attention = ST(13)

        init_module_weights(self.network, orthogonal_init)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        observations = observations.reshape(observations.shape[0], -1, 13)
        x = self.attention(observations)
        x = torch.cat((x, actions), dim=-1)
        q_values = torch.squeeze(self.network(x), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class CalQL:
    def __init__(
            self,
            critic_1,
            critic_1_optimizer,
            critic_2,
            critic_2_optimizer,
            actor,
            actor_optimizer,
            target_entropy: float,
            discount: float = 0.99,
            alpha_multiplier: float = 1.0,
            use_automatic_entropy_tuning: bool = True,
            backup_entropy: bool = False,
            policy_lr: bool = 3e-4,
            qf_lr: bool = 3e-4,
            soft_target_update_rate: float = 5e-3,
            bc_steps=100000,
            target_update_period: int = 1,
            cql_n_actions: int = 10,
            cql_importance_sample: bool = True,
            cql_lagrange: bool = False,
            cql_target_action_gap: float = -1.0,
            cql_temp: float = 1.0,
            cql_alpha: float = 5.0,
            cql_max_target_backup: bool = False,
            cql_clip_diff_min: float = -np.inf,
            cql_clip_diff_max: float = np.inf,
            device: str = "cpu",
    ):
        super().__init__()

        self.discount = discount
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_alpha = cql_alpha
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device

        self.total_it = 0

        self.critic_1 = critic_1
        self.critic_2 = critic_2

        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)

        self.actor = actor

        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )

        self._calibration_enabled = True
        self.total_it = 0

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)

    def switch_calibration(self):
        self._calibration_enabled = not self._calibration_enabled

    def _alpha_and_alpha_loss(self, observations: torch.Tensor, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                    self.log_alpha() * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            new_actions: torch.Tensor,
            alpha: torch.Tensor,
            log_pi: torch.Tensor,
    ) -> torch.Tensor:
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(observations, actions)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.critic_1(observations, new_actions),
                self.critic_2(observations, new_actions),
            )
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        return policy_loss

    def _q_loss(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            next_observations: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            mc_returns: torch.Tensor,
            alpha: torch.Tensor,
            log_dict: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1_predicted = self.critic_1(observations, actions)
        q2_predicted = self.critic_2(observations, actions)

        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.actor(
                next_observations, repeat=self.cql_n_actions
            )
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_critic_1(next_observations, new_next_actions),
                    self.target_critic_2(next_observations, new_next_actions),
                ),
                dim=-1,
            )
            next_log_pi = torch.gather(
                next_log_pi, -1, max_target_indices.unsqueeze(-1)
            ).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.actor(next_observations)
            target_q_values = torch.min(
                self.target_critic_1(next_observations, new_next_actions),
                self.target_critic_2(next_observations, new_next_actions),
            )

        if self.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        target_q_values = target_q_values.unsqueeze(-1)
        td_target = rewards + (1.0 - dones) * self.discount * target_q_values.detach()
        td_target = td_target.squeeze(-1)
        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        # CQL
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        cql_random_actions = actions.new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis = self.actor(
            observations, repeat=self.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis = self.actor(
            next_observations, repeat=self.cql_n_actions
        )
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )

        cql_q1_rand = self.critic_1(observations, cql_random_actions)
        cql_q2_rand = self.critic_2(observations, cql_random_actions)
        cql_q1_current_actions = self.critic_1(observations, cql_current_actions)
        cql_q2_current_actions = self.critic_2(observations, cql_current_actions)
        cql_q1_next_actions = self.critic_1(observations, cql_next_actions)
        cql_q2_next_actions = self.critic_2(observations, cql_next_actions)

        # Calibration
        lower_bounds = mc_returns.reshape(-1, 1).repeat(
            1, cql_q1_current_actions.shape[1]
        )

        num_vals = torch.sum(lower_bounds == lower_bounds)
        bound_rate_cql_q1_current_actions = (
                torch.sum(cql_q1_current_actions < lower_bounds) / num_vals
        )
        bound_rate_cql_q2_current_actions = (
                torch.sum(cql_q2_current_actions < lower_bounds) / num_vals
        )
        bound_rate_cql_q1_next_actions = (
                torch.sum(cql_q1_next_actions < lower_bounds) / num_vals
        )
        bound_rate_cql_q2_next_actions = (
                torch.sum(cql_q2_next_actions < lower_bounds) / num_vals
        )

        # """ Cal-QL: bound Q-values with MC return-to-go """
        if self._calibration_enabled:
            cql_q1_current_actions = torch.maximum(cql_q1_current_actions, lower_bounds)
            cql_q2_current_actions = torch.maximum(cql_q2_current_actions, lower_bounds)
            cql_q1_next_actions = torch.maximum(cql_q1_next_actions, lower_bounds)
            cql_q2_next_actions = torch.maximum(cql_q2_next_actions, lower_bounds)

        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                torch.unsqueeze(q1_predicted, 1),
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                torch.unsqueeze(q2_predicted, 1),
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5 ** action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.detach(),
                    cql_q1_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.detach(),
                    cql_q2_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf1_loss = (
                    alpha_prime
                    * self.cql_alpha
                    * (cql_qf1_diff - self.cql_target_action_gap)
            )
            cql_min_qf2_loss = (
                    alpha_prime
                    * self.cql_alpha
                    * (cql_qf2_diff - self.cql_target_action_gap)
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
            cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha
            alpha_prime_loss = observations.new_tensor(0.0)
            alpha_prime = observations.new_tensor(0.0)

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        log_dict.update(
            dict(
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha=alpha.item(),
                average_qf1=q1_predicted.mean().item(),
                average_qf2=q2_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
            )
        )

        log_dict.update(
            dict(
                cql_std_q1=cql_std_q1.mean().item(),
                cql_std_q2=cql_std_q2.mean().item(),
                cql_q1_rand=cql_q1_rand.mean().item(),
                cql_q2_rand=cql_q2_rand.mean().item(),
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
                cql_qf2_diff=cql_qf2_diff.mean().item(),
                cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                cql_q2_next_actions=cql_q2_next_actions.mean().item(),
                alpha_prime_loss=alpha_prime_loss.item(),
                alpha_prime=alpha_prime.item(),
                bound_rate_cql_q1_current_actions=bound_rate_cql_q1_current_actions.item(),  # noqa
                bound_rate_cql_q2_current_actions=bound_rate_cql_q2_current_actions.item(),  # noqa
                bound_rate_cql_q1_next_actions=bound_rate_cql_q1_next_actions.item(),
                bound_rate_cql_q2_next_actions=bound_rate_cql_q2_next_actions.item(),
            )
        )

        return qf_loss, alpha_prime, alpha_prime_loss

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            mc_returns,
        ) = batch
        self.total_it += 1

        new_actions, log_pi = self.actor(observations)

        alpha, alpha_loss = self._alpha_and_alpha_loss(observations, log_pi)

        """ Policy loss """
        policy_loss = self._policy_loss(
            observations, actions, new_actions, alpha, log_pi
        )

        log_dict = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
        )

        """ Q function loss """
        qf_loss, alpha_prime, alpha_prime_loss = self._q_loss(
            observations,
            actions,
            next_observations,
            rewards,
            dones,
            mc_returns,
            alpha,
            log_dict,
        )

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "critic1_target": self.target_critic_1.state_dict(),
            "critic2_target": self.target_critic_2.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "sac_log_alpha": self.log_alpha,
            "sac_log_alpha_optim": self.alpha_optimizer.state_dict(),
            "cql_log_alpha": self.log_alpha_prime,
            "cql_log_alpha_optim": self.alpha_prime_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.critic_1.load_state_dict(state_dict=state_dict["critic1"])
        self.critic_2.load_state_dict(state_dict=state_dict["critic2"])

        self.target_critic_1.load_state_dict(state_dict=state_dict["critic1_target"])
        self.target_critic_2.load_state_dict(state_dict=state_dict["critic2_target"])

        self.critic_1_optimizer.load_state_dict(
            state_dict=state_dict["critic_1_optimizer"]
        )
        self.critic_2_optimizer.load_state_dict(
            state_dict=state_dict["critic_2_optimizer"]
        )
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])

        self.log_alpha = state_dict["sac_log_alpha"]
        self.alpha_optimizer.load_state_dict(
            state_dict=state_dict["sac_log_alpha_optim"]
        )

        self.log_alpha_prime = state_dict["cql_log_alpha"]
        self.alpha_prime_optimizer.load_state_dict(
            state_dict=state_dict["cql_log_alpha_optim"]
        )
        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str,
                        default='D:\Program Files (x86)\IDE\JetBrains\PycharmProjects\CrowdNav-master_danren\crowd_nav\configs\env.config')
    args = parser.parse_args()
    # actor.eval()
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)

    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    state_dim = 65
    action_dim = 2

    file_path = "D:\Program Files (x86)\IDE\JetBrains\PycharmProjects\CrowdNav-master_danren\crowd_nav\Crowd_nav_5\Crowd_nav_5.hdf5"
    # file_path = "/hd_2t/czx/project/CrowdNav/crowd_nav/offline_data/Crowd_nav_6.hdf5"
    # file_path = "/hd_2t/fuhao2024/Yzc/NEW_cav/CrowdNav/Crowd_nav_5/Crowd_nav_5.hdf5"
    with h5py.File(file_path, 'r') as f:
        dataset = {k: np.array(v) for k, v in f.items()}

    # is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)
    batch_size_offline = int(config.batch_size * config.mixing_ratio)
    batch_size_online = config.batch_size - batch_size_offline


    mc_returns = get_return_to_go(dataset, env, config)
    dataset["mc_returns"] = np.array(mc_returns)

    offline_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    online_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    offline_buffer.load_d4rl_dataset(dataset)
    max_action = float(1)
    # max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)
    # Set seeds
    seed = config.seed
    set_seed(seed, env)
    # set_env_seed(eval_env, config.eval_seed)

    critic_1 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    critic_2 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    critic_1_optimizer = torch.optim.Adam(list(critic_1.parameters()), config.qf_lr)
    critic_2_optimizer = torch.optim.Adam(list(critic_2.parameters()), config.qf_lr)

    actor = TanhGaussianPolicy(
        state_dim,
        action_dim,
        max_action,
        orthogonal_init=config.orthogonal_init,
    ).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), config.policy_lr)
    robot.set_policy(actor)

    kwargs = {
        "critic_1": critic_1,
        "critic_2": critic_2,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2_optimizer": critic_2_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        # "target_entropy": -np.prod(env.action_space.shape).item(),
        "target_entropy": -2,
        "alpha_multiplier": config.alpha_multiplier,
        "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
        "backup_entropy": config.backup_entropy,
        "policy_lr": config.policy_lr,
        "qf_lr": config.qf_lr,
        "bc_steps": config.bc_steps,
        "target_update_period": config.target_update_period,
        "cql_n_actions": config.cql_n_actions,
        "cql_importance_sample": config.cql_importance_sample,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "cql_temp": config.cql_temp,
        "cql_alpha": config.cql_alpha,
        "cql_max_target_backup": config.cql_max_target_backup,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
    }

    print("---------------------------------------")
    print(f"Training Cal-QL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")
    now = datetime.now() + timedelta(hours=8)
    print("训练开始时间：", now.strftime("%Y-%m-%d %H:%M:%S"))
    # Initialize actor
    trainer = CalQL(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor



    print("Offline pretraining")
    episode_step = 0
    episode_return = 0
    # gamma = 1.0
    success = 0
    collision = 0
    run_time = 0
    avg_nav_time = 0
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    # 先 reset 环境
    cnt = 0
    rewards = 0
    phase = 'train'
    # config.offline_iterations
    for t in range(int(config.offline_iterations)+ int(config.online_iterations) ):   #  + int(config.online_iterations)
        if t == config.offline_iterations:
            print("Online tuning")
            trainer.switch_calibration()
            trainer.cql_alpha = config.cql_alpha_online
        online_log = {}
        if t >= config.offline_iterations:
            observation = env.reset(phase=phase, test_case=cnt)

            done = False
            while not done:

                joint_state = JointState(robot.get_full_state(), observation)
                state = to_np(transform(joint_state, config.device).view(1, -1).squeeze(0))
                actor.eval()
                raw_action = actor.act(state)  # np.array shape (2,)
                actor.train()
                action = ActionXY(raw_action[0], raw_action[1])
                # next_state, reward, done, info = env.step(action)
                observation, reward, done, info = env.step(action)

                next_joint_state = JointState(robot.get_full_state(), observation)
                next_state = to_np(transform(next_joint_state, config.device).view(1, -1).squeeze(0))

                online_buffer.add_transition(
                    state,
                    raw_action,
                    reward,
                    next_state,
                    done,
                )
                episode_return += reward



        if t < config.offline_iterations:
            batch = offline_buffer.sample(config.batch_size)
            batch = [b.to(config.device) for b in batch]
        else:
            offline_batch = offline_buffer.sample(batch_size_offline)
            offline_batch = [b.to(config.device) for b in offline_batch]

            # online buffer 不足时，不使用 online 数据
            if online_buffer._size < batch_size_online:
                online_batch = offline_buffer.sample(batch_size_online)
            else:
                online_batch = online_buffer.sample(batch_size_online)
            online_batch = [b.to(config.device) for b in online_batch]

            batch = [
                torch.vstack(tuple(b)).to(config.device)
                for b in zip(offline_batch, online_batch)
                ]
        # trainer.train(batch)
        log_dict = trainer.train(batch)
        log_dict["offline_iter" if t < config.offline_iterations else "online_iter"] = (
            t if t < config.offline_iterations else t - config.offline_iterations
        )
        log_dict.update(online_log)
        # wandb.log(log_dict, step=trainer.total_it)

        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            test_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )

            # # 保存 checkpoint
            # if config.checkpoints_path:
            #     torch.save(
            #         trainer.state_dict(),
            #         os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
            #     )

            # wandb 日志
            # wandb.log(eval_log, step=trainer.total_it)
    test_actor(
        env,
        actor,
        device=config.device,
        n_episodes=500,
        seed=config.seed,
    )
    list1 = np.array(list1)
    list2 = np.array(list2)
    list3 = np.array(list3)
    list4 = np.array(list4)
    # print(list1)
    # print(list2)
    # print(list3)
    # print(list4)
    ts_sec = int(time.time())
    pt_file_sec = f"rlhf_cql_{ts_sec}.pt"
    npz_file_sec = f"success_cql_{ts_sec}.npz"
    torch.save(trainer.state_dict(), os.path.join(config.checkpoints_path, pt_file_sec), )
    # np.savez(os.path.join('/hd_2t/fuhao2024/Yzc/table', npz_file_sec),
    #          success=list1, collision=list2, time=list3, reward=list4)


if __name__ == "__main__":
    train()
