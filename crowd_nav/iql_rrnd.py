# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import h5py
import argparse
import configparser
import logging


# ------------- Logging Utilities -------------
import logging as _logging  # alias to avoid shadowing
from datetime import datetime as _dt
import os as _os

import matplotlib.pyplot as plt


def setup_logger(log_dir: str, name: str = "iql"):
    """
    在 log_dir 下创建 train_eval.log（每次运行都会清空旧内容），并同时输出到控制台。
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 防止重复添加 handler

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path = os.path.join(log_dir, "train_eval.log")

    # 关键：mode='w' 覆盖写，确保每次运行先清空旧日志
    fh = logging.FileHandler(log_path, mode='w', encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ------------- End Logging Utilities -------------
import copy

import torchvision
from crowd_sim.envs.utils.info import ReachGoal, Collision, Timeout
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.robot import Robot

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
from crowd_sim.envs.utils.info import *
from crowd_nav.policy.self_attention import SelfAttention
from crowd_nav.policy.rnd import RND
TensorBatch = List[torch.Tensor]

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda:0"

    # env: str = "halfcheetah-medium-replay-v2"  # OpenAI gym environment name
    env: str = "CrowdSim-v0"
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(10000)  # How often (time steps) we evaluate
    n_episodes: int = 500  # How many episodes run during evaluation
    max_timesteps: int = int(100000)  # Max time steps to run environment
    checkpoints_path: Optional[
        str] = "D:\Program Files (x86)\IDE\JetBrains\PycharmProjects\CrowdNav-master_danren\crowd_nav\data\output_iql_group"  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL
    buffer_size: int = 1000000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    offline_batch_size: int = 256
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 0.5  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.3  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # Wandb logging
    project: str = "CORL"
    group: str = "IQL-D4RL"
    name: str = "IQL"

    # ====== RND & 离线→在线相关新增配置 ======
    offline_steps: int = 100000  # 只用于离线 IQL + RND 的更新次数（可以用原来的 max_timesteps 也行）
    online_episodes: int = 50000  # 在线微调执行多少个 episode
    online_batch_size: int = 4096  # 每次从在线 buffer 取多少条
    online_steps_per_update: int = 32  # 每收集够一个在线 batch 做几次更新
    online_buffer_max_size: int = 20000  # 在线 buffer 最大容量

    rnd_hidden_dim: int = 256
    rnd_output_dim: int = 32
    rnd_lr: float = 3e-4


    # === 下面两个是“保守在线更新”的关键参数 ===
    rnd_w_min: float = 0.5        # 反向 RND 权重阈值，越大越偏保守
    min_safe_online: int = 64     # 至少需要这么多“熟悉”的在线样本，才让在线数据参与更新

    # reward_model_path: str = "/hd_2t/czx/project/Clean-Offline-RLHF/rlhf/reward_model_logs/CrowdSim-v0/transformer/epoch_200_query_2000_len_15_seed_0/models/reward_model.pt"
    # reward_model_type: str = "transformer"
    # reward_model_type: str = "mlp"

    # def __post_init__(self):
    #     self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
    #     if self.checkpoints_path is not None:
    #         self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def clamp_xy(num, min, max):
    if num < min:
        num = min
    elif num > max:
        num = max
    else:
        pass
    return num

# class ReplayBuffer:
#     def __init__(
#             self,
#             state_dim: int,
#             action_dim: int,
#             buffer_size: int,
#             device: str = "cpu",
#     ):
#         self._buffer_size = buffer_size
#         self._pointer = 0
#         self._size = 0
#
#         self._states = torch.zeros(
#             (buffer_size, state_dim), dtype=torch.float32, device=device
#         )
#         self._actions = torch.zeros(
#             (buffer_size, action_dim), dtype=torch.float32, device=device
#         )
#         self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
#         self._next_states = torch.zeros(
#             (buffer_size, state_dim), dtype=torch.float32, device=device
#         )
#         self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
#         self._device = device
#
#     def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
#         return torch.tensor(data, dtype=torch.float32, device=self._device)
#
#     # Loads data in d4rl format, i.e. from Dict[str, np.array].
#     def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
#         if self._size != 0:
#             raise ValueError("Trying to load data into non-empty replay buffer")
#         n_transitions = data["observations"].shape[0]
#         if n_transitions > self._buffer_size:
#             raise ValueError(
#                 "Replay buffer is smaller than the dataset you are trying to load!"
#             )
#         self._states[:n_transitions] = self._to_tensor(data["observations"])
#         self._actions[:n_transitions] = self._to_tensor(data["actions"])
#         self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
#         self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
#         self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
#         self._size += n_transitions
#         self._pointer = min(self._size, n_transitions)
#
#         print(f"Dataset size: {n_transitions}")
#
#     def sample(self, batch_size: int) -> TensorBatch:
#         indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
#         states = self._states[indices]
#         actions = self._actions[indices]
#         rewards = self._rewards[indices]
#         next_states = self._next_states[indices]
#         dones = self._dones[indices]
#         # print(states.shape)
#         # print(actions.shape)
#         # print(rewards.shape)
#         # print(next_states.shape)
#         # print(dones.shape)
#         # print(dones)
#         return [states, actions, rewards, next_states, dones]
#
#     def add_transition(self):
#         # Use this method to add new data into the replay buffer during fine-tuning.
#         # I left it unimplemented since now we do not do fine-tuning.
#         raise NotImplementedError


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
        self._device = device
        self.rnd_max = 0
        self.rnd_min = 0
        # 数据存储
        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

        # 新增：存储重要性权重（反向 RND 权重）
        # 离线数据默认权重建议设为 1.0 (因为它们是分布内数据)
        self._weights = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset_old(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")

        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])

        # 初始化离线数据的权重为 1.0 (代表非常安全/熟悉)
        self._weights[:n_transitions] = 0

        self._size += n_transitions
        self._pointer = min(self._size, n_transitions) % self._buffer_size

        print(f"Dataset size: {n_transitions} loaded into unified buffer.")

    # 在你的 ReplayBuffer 类中 (通常在 buffer.py 或 iql_rrnd.py 中)

    # 在你的 ReplayBuffer 类中 (通常在 buffer.py 或 iql_rrnd.py 中)

    def load_d4rl_dataset(self, data: dict, weights=None):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")

        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])

        # 如果传入了权重则加载，否则默认为 1.0
        if weights is not None:
            self._weights[:n_transitions] = self._to_tensor(weights[..., None] if weights.ndim == 1 else weights)
        else:
            self._weights[:n_transitions] = 1.0

        self._size += n_transitions
        # 修正：虽然这里 pointer 可能回绕为 0，但 size 是正确的
        self._pointer = min(self._size, n_transitions) % self._buffer_size

        print(f"Dataset size: {n_transitions} loaded into unified buffer.")

    def sample(self, batch_size: int) -> TensorBatch:
        # [BUG FIX] 使用 self._size 而不是 min(size, pointer)
        # 只要 buffer 里有数据，就在 0 到 size 之间随机采
        indices = np.random.randint(0, self._size, size=batch_size)

        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]

        # 获取对应的权重
        weights = self._weights[indices]

        # 返回 6 个张量
        return [states, actions, rewards, next_states, dones, weights]





    # 实现添加单条数据
    def add_transition(self, state, action, reward, next_state, done, weight=1.0):
        # 确保输入是 tensor 且在正确的 device 上
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self._device)
        if not torch.is_tensor(action):
            action = torch.tensor(action, dtype=torch.float32, device=self._device)
        if not torch.is_tensor(next_state):
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self._device)

        # 写入 buffer
        self._states[self._pointer] = state
        self._actions[self._pointer] = action
        self._rewards[self._pointer] = float(reward)
        self._next_states[self._pointer] = next_state
        self._dones[self._pointer] = float(done)
        self._weights[self._pointer] = float(weight)

        # 更新指针和大小
        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)


# 在 iql_rrnd.py 中添加

class DualReplayBuffer:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            offline_buffer_size: int,
            online_buffer_size: int,
            device: str = "cpu",
    ):
        self.device = device

        # 1. 离线 Buffer (静态，只读)
        self.offline_buffer = ReplayBuffer(
            state_dim, action_dim, offline_buffer_size, device
        )

        # 2. 在线 Buffer (动态，循环写入)
        self.online_buffer = ReplayBuffer(
            state_dim, action_dim, online_buffer_size, device
        )

    def load_offline_dataset(self, data: Dict[str, np.ndarray], weights=None):
        """加载离线数据到 offline_buffer"""
        self.offline_buffer.load_d4rl_dataset(data, weights)

    def add_online_transition(self, state, action, reward, next_state, done, weight=1.0):
        """添加在线交互数据到 online_buffer"""
        self.online_buffer.add_transition(state, action, reward, next_state, done, weight)

    def sample(self, batch_size: int, online_ratio: float = 0.5) -> List[torch.Tensor]:
        """
        混合采样：分别从两个 buffer 采样，然后拼接
        online_ratio: 在线数据在 batch 中的占比 (0.0 ~ 1.0)
        """
        # 计算各自需要采样的数量
        n_online = int(batch_size * online_ratio)
        n_offline = batch_size - n_online

        # 确保在线 buffer 有足够数据，否则全采离线
        if self.online_buffer._size < n_online:
            return self.offline_buffer.sample(batch_size)

        # 分别采样
        # 假设 sample 返回 [states, actions, rewards, next_states, dones, weights]
        offline_batch = self.offline_buffer.sample(n_offline)
        online_batch = self.online_buffer.sample(n_online)

        # 拼接数据 (Concatenate)
        combined_batch = []
        for i in range(len(offline_batch)):
            # dim=0 是 batch 维度
            combined = torch.cat([offline_batch[i], online_batch[i]], dim=0)
            combined_batch.append(combined)

        return combined_batch





def set_seed(
        seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        # env.action_space.seed(seed)
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



cnt = 20000


@torch.no_grad()
def eval_actor(
        env: gym.Env, actor: nn.Module, list1: List, list2: List, list3: List, list4: List, device: str,
        n_episodes: int, seed: int
        , logger: Optional[logging.Logger] = None, global_step: Optional[int] = None):
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

    metrics = {
        "success": success,
        "success_rate": success / n_episodes if n_episodes else 0.0,
        "collision": collision,
        "timeout": run_time,
        "avg_nav_time": avg_nav_time,
        "avg_reward": avg_reward,
        "episodes": n_episodes,
        "step": global_step,
    }
    # 列表记录（若提供）
    try:
        list1.append(metrics["success_rate"])
        list2.append(collision / n_episodes if n_episodes else 0.0)
        list3.append(avg_nav_time)
        list4.append(avg_reward)
    except Exception:
        pass
    if logger is not None:
        try:
            logger.info(
                "[EVAL] step=%s | succ=%d(%.3f) | coll=%d | timeout=%d | nav_time=%.6f | reward=%.6f | episodes=%d",
                str(global_step), success, metrics["success_rate"], collision, run_time, avg_nav_time, avg_reward,
                n_episodes
            )
        except Exception:
            pass
    actor.train()

    # print(
    #     f"成功次数为{success} 成功率为{success / n_episodes} 碰撞次数{collision} 超时次数{run_time} 平均导航时间为{avg_nav_time} 平均奖励为{avg_reward} ")
    list1.append(success / n_episodes)
    list2.append(collision / n_episodes)
    list3.append(avg_nav_time)
    list4.append(avg_reward)
    actor.train()
    return metrics



@torch.no_grad()
def test_actor(
        env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
        , logger: Optional[logging.Logger] = None):
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str,
                        default="D:\Program Files (x86)\IDE\JetBrains\PycharmProjects\CrowdNav-master_danren\crowd_nav\configs\env.config")
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
        # env.render(mode='traj')
        # plt.show()

        avg_reward += rewards
    avg_nav_time = avg_nav_time / success
    avg_reward = avg_reward / n_episodes

    metrics = {
        "success": success,
        "success_rate": success / n_episodes if n_episodes else 0.0,
        "collision": collision,
        "timeout": run_time,
        "avg_nav_time": avg_nav_time,
        "avg_reward": avg_reward,
        "episodes": n_episodes
    }


    if logger is not None:
        try:
            logger.info(
                "[TEST] succ=%d(%.3f) | coll=%d | timeout=%d | nav_time=%.6f | reward=%.6f | episodes=%d",
                success, metrics["success_rate"], collision, run_time, avg_nav_time, avg_reward, n_episodes
            )
        except Exception:
            pass
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


def return_reward_range(dataset, max_episode_steps):
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


# 辅助函数：加权的不对称 L2 Loss
def weighted_asymmetric_l2_loss(u: torch.Tensor, tau: float, weights: torch.Tensor) -> torch.Tensor:
    # weights shape: (batch, 1)
    element_wise_loss = torch.abs(tau - (u < 0).float()) * u ** 2
    return torch.mean(weights * element_wise_loss)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
            self,
            dims,
            activation_fn: Callable[[], nn.Module] = nn.ReLU,
            output_activation_fn: Callable[[], nn.Module] = None,
            squeeze_output: bool = False,
            dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            act_dim: int,
            max_action: float,
            hidden_dim: int = 256,
            n_hidden: int = 2,
            dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [56, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action
        self.kinematics = 'holonomic'
        self.multiagent_training = True
        self.attention = SelfAttention()

    def forward(self, obs: torch.Tensor) -> Normal:
        x = self.attention(obs)
        mean = self.net(x)
        # print(mean.shape)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cuda"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            act_dim: int,
            max_action: float,
            hidden_dim: int = 256,
            n_hidden: int = 2,
            dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action
        self.kinematics = 'holonomic'
        self.multiagent_training = True

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [58, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)
        self.attention = SelfAttention()

    def both(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.attention(state)
        x = torch.cat((x, action), dim=1)
        # sa = torch.cat([state, action], 1)
        return self.q1(x), self.q2(x)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [56, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)
        self.attention = SelfAttention()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.attention(state)
        return self.v(x)


class ImplicitQLearning:
    def __init__(
            self,
            max_action: float,
            actor: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            q_network: nn.Module,
            q_optimizer: torch.optim.Optimizer,
            v_network: nn.Module,
            v_optimizer: torch.optim.Optimizer,
            iql_tau: float = 0.7,  # 0.7
            beta: float = 3.0,     # 3
            max_steps: int = 1000000,
            discount: float = 0.99,
            tau: float = 0.005,
            device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau
        self.total_it = 0
        self.device = device

    # 辅助函数：加权的不对称 L2 Loss
    # def weighted_asymmetric_l2_loss(u: torch.Tensor, tau: float, weights: torch.Tensor) -> torch.Tensor:
    #     # weights shape: (batch, 1)
    #     element_wise_loss = torch.abs(tau - (u < 0).float()) * u ** 2
    #     return torch.mean(weights * element_wise_loss)

    # 1. 修改 Value Network 更新
    def _update_v(self, observations, actions, weights, log_dict) -> torch.Tensor:
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v

        # [修改] 使用加权 Loss
        # weights shape: (B, 1), element_wise_loss shape: (B, 1) -> mean scalar
        element_wise_loss = torch.abs(self.iql_tau - (adv < 0).float()) * adv ** 2
        v_loss = torch.mean(weights * element_wise_loss)

        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    # 2. 修改 Q Network 更新
    def _update_q(
            self,
            next_v: torch.Tensor,
            observations: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            terminals: torch.Tensor,
            weights: torch.Tensor,  # [新增输入]
            log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)

        # [修改] 加权 MSE Loss
        q_loss_total = 0
        for q in qs:
            # (q - targets)^2 是 element-wise 的，乘以 weights 后再求均值
            q_loss_total += torch.mean(weights * (q - targets) ** 2)

        q_loss = q_loss_total / len(qs)

        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        soft_update(self.q_target, self.qf, self.tau)

    # 3. 修改 Policy 更新
    def _update_policy_without_group(
            self,
            adv: torch.Tensor,
            observations: torch.Tensor,
            actions: torch.Tensor,
            weights: torch.Tensor,  # [新增输入]
            log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)

        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError

        # [修改] 加权 Policy Loss
        # weights: [B, 1] -> squeeze -> [B]
        # exp_adv: [B], bc_losses: [B]
        weights_squeeze = weights.squeeze(-1)
        policy_loss = torch.mean(weights_squeeze * exp_adv * bc_losses)

        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    # 4. 修改 Train 主入口
    def train(self, batch: TensorBatch,weights) -> Dict[str, float]:
        self.total_it += 1

        # [修改] 解包 6 个元素
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            old_weights,
        ) = batch

        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)

        # 传入 weights
        adv = self._update_v(observations, actions, weights, log_dict)

        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)

        # 传入 weights
        self._update_q(next_v, observations, actions, rewards, dones, weights, log_dict)

        # 传入 weights
        self._update_policy_without_group(adv, observations, actions, weights, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]

# def sample_online_batch(online_buffer, batch_size, device):
#     """
#     online_buffer: list of (state, action, reward, next_state, done, weight)
#                    其中 weight 已经是 “反向 RND 权重”：见过多 -> 大，没见过 -> 小
#     返回: TensorBatch [states, actions, rewards, next_states, dones]
#     """
#     n = len(online_buffer)
#     assert n >= batch_size
#
#     weights = np.array([tr[5] for tr in online_buffer], dtype=np.float32)
#     probs = weights / (weights.sum() + 1e-8)
#
#     idx = np.random.choice(n, size=batch_size, replace=True, p=probs)
#
#     s = np.array([online_buffer[i][0] for i in idx], dtype=np.float32)
#     a = np.array([online_buffer[i][1] for i in idx], dtype=np.float32)
#     r = np.array([online_buffer[i][2] for i in idx], dtype=np.float32).reshape(-1, 1)
#     ns = np.array([online_buffer[i][3] for i in idx], dtype=np.float32)
#     d = np.array([online_buffer[i][4] for i in idx], dtype=np.float32).reshape(-1, 1)
#
#     states = torch.tensor(s, device=device)
#     actions = torch.tensor(a, device=device)
#     rewards = torch.tensor(r, device=device)
#     next_states = torch.tensor(ns, device=device)
#     dones = torch.tensor(d, device=device)
#
#     return [states, actions, rewards, next_states, dones]


def sample_online_batch(
        online_buffer: List,  # 假设是 list of tuples
        batch_size: int,
        device,
        w_min: float = 0.15,  # 建议默认值：0.15 (对应 norm_error <= 1.0，即不超过离线最大误差)
        min_safe_num: int = 10,  # 至少要有 10 个安全样本才开始利用在线数据
):
    """
    从在线 buffer 中采样。
    由于权重现在已经归一化到 (0, 1] 且区分度很大：
    1. 即使不设 w_min，加权采样也会自动偏向安全样本。
    2. 设 w_min 可以作为硬性过滤，剔除那些极端危险（Error 远超离线分布）的离群点。
    """
    n = len(online_buffer)
    if n == 0:
        return None

    # 1. 提取权重 (优化：如果 buffer 很大，建议在 add 时就维护一个独立的 weights numpy array，不要每次 list comprehension)
    # 假设 online_buffer 里的结构是 (s, a, r, ns, d, weight)
    weights = np.array([tr[5] for tr in online_buffer], dtype=np.float32)

    # 2. 确定候选索引
    # 逻辑：先用 w_min 剔除"完全不可接受"的样本，剩下的样本中按权重概率采样

    # 筛选出"及格"的样本
    # w_min = 0.16 意味着允许 norm_error 高达 1.0 (即离线数据的最大误差边界)
    # w_min = 0.50 意味着要求 norm_error < 0.2 (非常安全)
    safe_mask = weights >= w_min
    safe_indices = np.where(safe_mask)[0]

    # 如果安全样本太少，直接放弃本次在线采样，退回只用离线数据
    if len(safe_indices) < min_safe_num:
        return None

    # 3. 在及格样本中，进行"加权"采样
    # 这样既保证了不采离谱的样本，又保证了在安全样本里优先采更安全的
    safe_weights = weights[safe_indices]

    # 加上 1e-8 防止除零
    probs = safe_weights / (safe_weights.sum() + 1e-8)

    # 4. 执行采样
    # 注意：如果安全样本数 < batch_size，允许重复采样 (replace=True) 或者只采实际数量
    idx_in_safe = np.random.choice(len(safe_indices), size=batch_size, replace=True, p=probs)
    idx = safe_indices[idx_in_safe]

    # 5. 组装 Batch
    # (这里可以用 zip(*) 来稍微加速解包，或者直接像你原来那样写)
    batch_data = [online_buffer[i] for i in idx]

    # 拆分数据
    states = torch.tensor(np.array([t[0] for t in batch_data]), device=device, dtype=torch.float32)
    actions = torch.tensor(np.array([t[1] for t in batch_data]), device=device, dtype=torch.float32)
    rewards = torch.tensor(np.array([t[2] for t in batch_data]), device=device, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(np.array([t[3] for t in batch_data]), device=device, dtype=torch.float32)
    dones = torch.tensor(np.array([t[4] for t in batch_data]), device=device, dtype=torch.float32).unsqueeze(1)

    return [states, actions, rewards, next_states, dones]






def sample_online_batch_uniform(online_buffer, batch_size, device):
    """
    不使用 RND 权重，在线数据一律等权重，均匀采样。
    online_buffer: list of (state, action, reward, next_state, done)
    """
    n = len(online_buffer)
    assert n >= batch_size, "online_buffer 中样本数量不足"

    idx = np.random.choice(n, size=batch_size, replace=True)

    s = np.array([online_buffer[i][0] for i in idx], dtype=np.float32)
    a = np.array([online_buffer[i][1] for i in idx], dtype=np.float32)
    r = np.array([online_buffer[i][2] for i in idx], dtype=np.float32).reshape(-1, 1)
    ns = np.array([online_buffer[i][3] for i in idx], dtype=np.float32)
    d = np.array([online_buffer[i][4] for i in idx], dtype=np.float32).reshape(-1, 1)

    states = torch.tensor(s, device=device)
    actions = torch.tensor(a, device=device)
    rewards = torch.tensor(r, device=device)
    next_states = torch.tensor(ns, device=device)
    dones = torch.tensor(d, device=device)

    return [states, actions, rewards, next_states, dones]



def online_finetune_with_reverse_rnd_(
    trainer,
    rnd,
    replay_buffer,   # 离线 buffer，用来和在线数据混合训练
    config: TrainConfig,
    logger: Optional[logging.Logger] = None,
):
    """
    trainer: 已经在离线阶段训练好的 IQL（包含 actor）
    rnd:    离线阶段训练好的 RND
    """

    # 复用 eval_actor 里那套环境构造
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument(
        '--env_config',
        type=str,
        default='D:\\Program Files (x86)\\IDE\\JetBrains\\PycharmProjects\\CrowdNav-master_danren\\crowd_nav\\configs\\env.config'
    )
    args, _ = parser.parse_known_args()

    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    robot.set_policy(trainer.actor)

    online_buffer = []
    device = config.device

    for ep in range(config.online_episodes):
        # 这里用 phase='train'，你也可以根据自己的环境改
        observation = env.reset(phase='train')
        done = False

        # ===== 在线阶段每个 episode 的统计量 =====
        success = 0
        collision = 0
        timeout = 0
        episode_reward = 0.0
        gamma = 1.0
        nav_time = 0.0
        info = None

        while not done:
            # 从 CrowdSim 状态转换成 65 维 state（和离线 hdf5 一致）
            joint_state = JointState(robot.get_full_state(), observation)
            state_vec = to_np(transform(joint_state, device).view(1, -1).squeeze(0))  # (65,)

            # 用当前策略选动作
            action = trainer.actor.act(state_vec, device=device)  # (2,)
            action_xy = ActionXY(action[0], action[1])

            # 加点噪声
            vx = action_xy.vx + np.random.normal(0, 0.1)
            vx = clamp_xy(vx, -1, 1)
            vy = action_xy.vy + np.random.normal(0, 0.1)
            vy = clamp_xy(vy, -1, 1)
            action_xy = ActionXY(vx, vy)

            next_obs, reward, done, info = env.step(action_xy)

            # 折扣奖励，和 eval_actor/test_actor 一致
            episode_reward += reward * gamma
            gamma *= 0.99

            next_joint_state = JointState(robot.get_full_state(), next_obs)
            next_state_vec = to_np(transform(next_joint_state, device).view(1, -1).squeeze(0))

            # ========= 关键：反向 RND 权重 =========
            novelty = rnd.get_intrinsic_reward(state_vec)  # 标量
            reverse_weight = 1.0 / (1.0 + novelty)        # 简单反函数，数值范围 (0,1]

            online_buffer.append(
                (state_vec, action, reward, next_state_vec, float(done), reverse_weight)
            )

            # 在线继续训练 RND，使“经常访问”的区域变成熟悉状态
            state_tensor = torch.tensor(state_vec, device=device, dtype=torch.float32).unsqueeze(0)
            rnd.train_predictor(state_tensor)

            observation = next_obs

            # ====== 每累积到一定量在线数据，就做一次 offline+online 混合训练 ======
            if len(online_buffer) >= config.online_batch_size:
                # 1) 从离线 buffer 随机采样一批
                offline_batch = replay_buffer.sample(config.batch_size)
                offline_batch = [b.to(device) for b in offline_batch]

                # 2) 从在线 buffer 用“反向 RND 权重”加权采样一批
                online_batch = sample_online_batch(
                    online_buffer,
                    config.online_batch_size,
                    device
                )

                # 3) 拼成一个大 batch，一起训练（最简单的混合方式）
                obs = torch.cat([offline_batch[0], online_batch[0]], dim=0)
                act = torch.cat([offline_batch[1], online_batch[1]], dim=0)
                rew = torch.cat([offline_batch[2], online_batch[2]], dim=0)
                nxt = torch.cat([offline_batch[3], online_batch[3]], dim=0)
                don = torch.cat([offline_batch[4], online_batch[4]], dim=0)

                mixed_batch = [obs, act, rew, nxt, don]

                for _ in range(config.online_steps_per_update):
                    trainer.train(mixed_batch)

                # 控制在线 buffer 大小，防止无限增长
                if len(online_buffer) > config.online_buffer_max_size:
                    online_buffer = online_buffer[-config.online_buffer_max_size:]

        # ===== 一个 episode 结束后，根据 info 统计结果，并写到日志 =====
        if isinstance(info, ReachGoal):
            success = 1
            nav_time = env.global_time
        elif isinstance(info, Collision):
            collision = 1
        elif isinstance(info, Timeout):
            timeout = 1

        success_rate = float(success)  # 每个 ep 只有一条轨迹
        avg_nav_time = nav_time
        avg_reward = episode_reward

        if logger is not None:
            # 格式尽量和你 offline 的 [TEST] 类似
            logger.info(
                "[ONLINE] ep=%d/%d | succ=%d(%.3f) | coll=%d | timeout=%d | nav_time=%.6f | reward=%.6f | episodes=%d | buffer=%d",
                ep + 1,
                config.online_episodes,
                success,
                success_rate,
                collision,
                timeout,
                avg_nav_time,
                avg_reward,
                1,
                len(online_buffer),
            )

        print(
            f"[Online] Episode {ep + 1}/{config.online_episodes} done "
            f"| succ={success} coll={collision} timeout={timeout} "
            f"| nav_time={avg_nav_time:.3f} reward={avg_reward:.3f} "
            f"| buffer size={len(online_buffer)}"
        )








# 可以将TrainConfig里面的参数映射到train方法里面



def online_finetune_weighted_loss(
        trainer,
        rnd,
        dual_buffer: DualReplayBuffer,
        config: TrainConfig,
        logger: Optional[logging.Logger] = None,
):
    """
    策略：
    1. 采样：使用 replay_buffer.sample() 进行全随机采样（离线+在线混合，不挑食）。
    2. 权重：计算 RND weight，并在 Loss 计算时使用它。
       熟悉的数据（离线/低Novelty） -> Weight ≈ 1.0 -> 正常更新。
       陌生的数据（高Novelty）     -> Weight -> 0.0 -> 几乎不更新（忽略该梯度）。
    """

    # ... (环境初始化代码，同前) ...
    parser = argparse.ArgumentParser('Parse configuration file')
    args, _ = parser.parse_known_args()
    # 假设 args.env_config 已经设置好
    env_config = configparser.RawConfigParser()
    env_config.read(
        "D:\\Program Files (x86)\\IDE\\JetBrains\\PycharmProjects\\CrowdNav-master_danren\\crowd_nav\\configs\\env.config")
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    robot.set_policy(trainer.actor)
    device = config.device

    steps_collected = 0
    online_sampling_ratio = 0.5
    # [新增] 临时 Buffer 用于存最新的 Online 数据给 RND 训练
    online_states_queue = []

    for ep in range(config.online_episodes):
        observation = env.reset(phase="train")
        done = False
        episode_reward = 0.0
        gamma = 1.0

        # 统计量
        success = 0
        collision = 0
        timeout = 0
        episode_reward = 0.0
        nav_time = 0.0
        info = None

        while not done:
            # --- 1. 状态处理 ---
            joint_state = JointState(robot.get_full_state(), observation)
            state_vec = to_np(transform(joint_state, device).view(1, -1).squeeze(0))

            # --- 2. 动作选择 (带一点噪声探索) ---
            action = trainer.actor.act(state_vec, device=device)
            action_xy = ActionXY(action[0], action[1])

            # 简单的探索噪声
            vx = clamp_xy(action_xy.vx + np.random.normal(0, 0.1), -1, 1)
            vy = clamp_xy(action_xy.vy + np.random.normal(0, 0.1), -1, 1)
            action_np = np.array([vx, vy])

            # --- 3. 环境交互 ---
            next_obs, reward, done, info = env.step(ActionXY(vx, vy))
            episode_reward += reward * gamma
            gamma *= config.discount

            next_joint_state = JointState(robot.get_full_state(), next_obs)
            next_state_vec = to_np(transform(next_joint_state, device).view(1, -1).squeeze(0))

            # --- 4. 计算 Loss 权重 (关键步骤) ---
            std_novelty = rnd.get_intrinsic_reward(state_vec, update_stats=True)
            # novelty = rnd.get_intrinsic_reward(state_vec)

            # 使用 Sigmoid 变体进行平滑映射
            # k 是敏感度系数，NeurIPS 论文中建议调节此参数
            k = 1.0
            # 这种映射保证了：熟悉的数据权重~1，极其陌生的数据权重~0.5或更低
            adaptive_weight = 1.0 / (1.0 + np.exp(k * std_novelty))

            # 归一化处理 (可选，但推荐)
            # 使用离线数据的统计量将 novelty 映射到合理范围，防止权重过小
            # 如果不归一化，novelty 很大时 weight 会接近 0，网络可能完全不学在线数据
            # norm_novelty = (novelty - replay_buffer.rnd_min) / (replay_buffer.rnd_max - replay_buffer.rnd_min + 1e-8)
            # norm_novelty = max(0.0, norm_novelty)  # 截断下限
            #
            # k = 0.0  # 调节系数，k越大，对新颖样本的惩罚越重
            # loss_weight = 1.0 / (1.0 + k * norm_novelty)

            # --- 5. 存入 Buffer ---
            # replay_buffer.add_transition(
            #     state_vec,
            #     action_np,
            #     reward,
            #     next_state_vec,
            #     float(done),
            #     weight=adaptive_weight
            # )
            dual_buffer.add_online_transition(
                state_vec,
                action_np,
                reward,
                next_state_vec,
                float(done),
                weight=adaptive_weight
            )
            # online_states_queue.append(state_vec)

            # (可选) 在线训练 RND，让它逐渐适应新环境
            # rnd.train_predictor(torch.tensor(state_vec, device=device).unsqueeze(0))

            observation = next_obs
            steps_collected += 1



        # Episode 结束统计
        if isinstance(info, ReachGoal):
            success = 1
            nav_time = env.global_time
        elif isinstance(info, Collision):
            collision = 1
        elif isinstance(info, Timeout):
            timeout = 1

       # --- 6. 训练 (随机采样 + 加权更新) ---
       #  if steps_collected % config.online_steps_per_update == 0:
            # 使用标准的随机采样 (Sample Uniformly)
            # 因为 sample() 现在返回了 weights，所以 trainer 会自动用到它
            # train_batch = replay_buffer.sample(config.batch_size)
        # 在 ReplayBuffer 中，你不再需要存储 'weights'，或者存了也不用它
        # batch 采样出来: states, actions, rewards, next_states, dones
        batch = dual_buffer.sample(batch_size=config.batch_size, online_ratio=online_sampling_ratio)
        states = batch[0].to(device)

        # --- [关键步骤] 实时计算当前 Batch 的权重 ---
        with torch.no_grad():
            # 1. 现场过一遍 RND，拿到最新的 Error
            # 注意：这里 update_stats=False，因为我们不想让 replay 的数据干扰
            # 那些用于标准化 (RunningMeanStd) 的统计量，统计量应该只在采集(collect)时更新
            raw_error = rnd.get_raw_error_batch(states)

            # 2. 使用当前的统计量进行归一化 (利用之前实现的 rms)
            # 这样能保证权重的尺度是适应当前网络水平的
            mean = rnd.rms.mean
            std = np.sqrt(rnd.rms.var) + 1e-8
            norm_error = (raw_error - mean) / std

            # 3. 映射为权重 (Sigmoid 或其他公式)
            # 这样算出来的 weight 才是“当下”真实的置信度
            k = 1.0
            current_weights = 1.0 / (1.0 + torch.exp(k * norm_error))
            # current_weights = 1.0 / 1.0 + k * norm_error
            # 确保维度匹配 (batch_size, 1)
            if current_weights.dim() == 1:
                current_weights = current_weights.unsqueeze(1)

        # --- 传入 Trainer ---
        # 你需要修改 trainer.train 接口，让它接受外部传入的 weights
        # 而不是从 batch 里面拆包 weights
        trainer.train(batch, weights=current_weights)


        # --- 4. 自适应更新 RND 网络 (每 N 步) ---
        if ep % 100 == 0 and len(online_states_queue) > config.online_batch_size:
            # 采样在线数据
            recent_online = np.array(online_states_queue[-config.online_batch_size:])
            online_tensor = torch.tensor(recent_online, device=config.device, dtype=torch.float32)

            # [关键] 采样离线数据 (Anchor)
            # 这对应了 FamO2O/APL 中“不忘本”的思想
            offline_batch = dual_buffer.offline_buffer.sample(config.online_batch_size)
            offline_tensor = offline_batch[0].to(config.device)  # 只取 state

            # 混合更新
            rnd.train_conflict_aware(online_tensor, offline_tensor)

        # 维护 queue 大小
        online_states_queue = online_states_queue[-6000:]




        if logger is not None:
            logger.info(
                "[ONLINE_GROUP_RND] ep=%d/%d | succ=%d | coll=%d | timeout=%d | nav_time=%.4f | reward=%.4f | buffer_size=%d",
                ep + 1,
                config.online_episodes,
                success,
                collision,
                timeout,
                nav_time,
                episode_reward,
                dual_buffer.online_buffer._size,
            )

        # ... (日志统计部分同前) ...
        print(f"Ep {ep} | Reward: {episode_reward:.3f} | Last Weight: {adaptive_weight:.4f}")




@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)

    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]

    state_dim = 65
    action_dim = 2

    # dataset = d4rl.qlearning_dataset(env)

    file_path = "D:\Program Files (x86)\IDE\JetBrains\PycharmProjects\CrowdNav-master_danren\crowd_nav\Crowd_nav_5\Crowd_nav_5.hdf5"
    with h5py.File(file_path, 'r') as f:
        dataset = {k: np.array(v) for k, v in f.items()}

    print("最小值:", dataset["rewards"].min(), "最大值:", dataset["rewards"].max())

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset_old(dataset)

    # ====== 新增：初始化 RND，用离线数据训练它 ======
    rnd = RND(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config.rnd_hidden_dim,
        output_dim=config.rnd_output_dim,
        lr=config.rnd_lr,
        device=config.device
    )

    # max_action = float(env.action_space.high[0])
    max_action = float(1)

    # 保存配置信息的
    # if config.checkpoints_path is not None:
    #     print(f"Checkpoints path: {config.checkpoints_path}")
    #     os.makedirs(config.checkpoints_path, exist_ok=True)
    #     with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
    #         pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}, Device: {config.device}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    # 初始化日志
    os.makedirs(config.checkpoints_path, exist_ok=True)
    logger = setup_logger(config.checkpoints_path, name="iql")
    logger.info("Start training IQL | env=%s | seed=%d | device=%s", config.env, config.seed, config.device)

    # if config.load_model != "":
    #     policy_file = Path(config.load_model)
    #     trainer.load_state_dict(torch.load(policy_file))
    #     actor = trainer.actor

    # wandb_init(asdict(config))
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    evaluations = []
    for t in range(int(config.max_timesteps)):

        batch = replay_buffer.sample(config.batch_size)
        # print(batch)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)


        # 用同一批观测训练 RND（让“离线分布”上的 state 误差变小）
        observations = batch[0]  # [B, state_dim]
        actions = batch[1]       # [B, action_dim]
        rnd.train_predictor(observations,actions)

        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            metrics = eval_actor(env, actor, list1, list2, list3, list4, device=config.device,
                                 n_episodes=config.n_episodes, seed=config.seed, logger=logger, global_step=t + 1)

    test_actor(env, actor, device=config.device, n_episodes=500, seed=config.seed, logger=logger, )
    list1 = np.array(list1)
    list2 = np.array(list2)
    list3 = np.array(list3)
    list4 = np.array(list4)
    print(list1)
    print(list2)
    print(list3)
    print(list4)
    logger.info("Training finished. Saving model...")
    # torch.save(trainer.state_dict(), os.path.join(config.checkpoints_path, f"iql.pt"), )
    # np.savez('D:\Program Files (x86)\IDE\JetBrains\PycharmProjects\CrowdNav-master_danren\crowd_nav\data\output_iql\success2.pth', success = list1, collision = list2, time = list3, reward = list4)

    # ========= 离线阶段收尾：保存所有在线微调会用到的东西 =========
    logger.info("Offline training finished. Saving offline checkpoints...")

    # 1) 保存完整 IQL trainer（含 actor/q/v + optimizer + lr_schedule + total_it）
    offline_iql_path = os.path.join(config.checkpoints_path, "iql_offline.pt")
    torch.save(trainer.state_dict(), offline_iql_path)

    # 2) 单独保存一份 actor，方便之后直接加载策略部署用
    actor_offline_path = os.path.join(config.checkpoints_path, "actor_offline.pt")
    torch.save(trainer.actor.state_dict(), actor_offline_path)

    # 3) 保存 RND 网络 + 其优化器（在线阶段要继续训 RND）
    rnd_offline_path = os.path.join(config.checkpoints_path, "rnd_offline.pt")
    torch.save(
        {
            "rnd_state": rnd.state_dict(),
            "rnd_optimizer": rnd.optimizer.state_dict(),
        },
        rnd_offline_path,
    )

    logger.info("Saved offline IQL to %s", offline_iql_path)
    logger.info("Saved offline actor to %s", actor_offline_path)
    logger.info("Saved offline RND to %s", rnd_offline_path)



    # # ========== 第二阶段：在线微调（反向 RND 加权采样） ==========
    # online_finetune_with_reverse_rnd(
    #     trainer=trainer,
    #     rnd=rnd,
    #     replay_buffer=replay_buffer,
    #     config=config,
    #     logger=logger,
    #
    # )
    #
    # # 在线阶段结束后，再做一次大规模 test
    # test_actor(
    #     env,
    #     trainer.actor,
    #     device=config.device,
    #     n_episodes=500,
    #     seed=config.seed,
    # )
    #
    # # 原来的统计与保存不变
    # list1 = np.array(list1)
    # list2 = np.array(list2)
    # list3 = np.array(list3)
    # list4 = np.array(list4)
    # print(list1)
    # print(list2)
    # print(list3)
    # print(list4)
    # torch.save(trainer.state_dict(), os.path.join(config.checkpoints_path, f"iql_rnd_offline_online.pt"))
    # np.savez(
    #     'D:\Program Files (x86)\IDE\JetBrains\PycharmProjects\CrowdNav-master_danren\crowd_nav\data\output_iql\success2.pth',
    #     success=list1, collision=list2, time=list3, reward=list4)
if __name__ == "__main__":
    train()