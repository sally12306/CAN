# source: Adapted from IQL implementation for AWAC
# https://arxiv.org/abs/2006.09359
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
import matplotlib.pyplot as plt
# ------------- Logging Utilities -------------
import logging as _logging
from datetime import datetime as _dt
import os as _os


def setup_logger(log_dir: str, name: str = "awac"):
    """
    在 log_dir 下创建 train_eval_awac.log，并同时输出到控制台。
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path = os.path.join(log_dir, "train_eval_awac.log")

    # mode='w' 覆盖写
    fh = logging.FileHandler(log_path, mode='w', encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ------------- Imports for CrowdSim -------------
from crowd_sim.envs.utils.info import ReachGoal, Collision, Timeout
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.self_attention import SelfAttention

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

TensorBatch = List[torch.Tensor]

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda:0"
    env: str = "CrowdSim-v0"
    seed: int = 0
    eval_freq: int = int(5000)
    n_episodes: int = 500

    # 离线预训练步数
    max_timesteps: int = int(100000)

    # Checkpoint 保存路径
    checkpoints_path: str = "D:\\Program Files (x86)\\IDE\\JetBrains\\PycharmProjects\\CrowdNav-master_danren\\crowd_nav\\data\\output_awac"

    # AWAC Parameters
    buffer_size: int = 2000000
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005

    # AWAC Temperature (beta):
    # 越小 -> 越接近 BC (保守)
    # 越大 -> 越倾向于高 Advantage 动作 (激进)
    # 建议范围: 0.3 ~ 3.0. 如果使用优势归一化，建议 1.0 ~ 2.0
    beta: float = 1.0

    normalize: bool = True

    # Learning Rates
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4

    actor_dropout: Optional[float] = None
    iql_deterministic: bool = False  # AWAC 也可以用确定性策略，但通常高斯策略效果更好

    # Online Fine-tuning related (Buffer configs)
    online_buffer_max_size: int = 100000

    project: str = "CrowdNav-AWAC"
    group: str = "Offline-Training"
    name: str = "AWAC"


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


# ------------- Replay Buffer -------------
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

        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset!")

        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])

        self._size += n_transitions
        self._pointer = min(self._size, n_transitions) % self._buffer_size
        print(f"Dataset size: {n_transitions} loaded.")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self, state, action, reward, next_state, done):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self._device)
        if not torch.is_tensor(action):
            action = torch.tensor(action, dtype=torch.float32, device=self._device)
        if not torch.is_tensor(next_state):
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self._device)

        self._states[self._pointer] = state
        self._actions[self._pointer] = action
        self._rewards[self._pointer] = float(reward)
        self._next_states[self._pointer] = next_state
        self._dones[self._pointer] = float(done)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)


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
        self.offline_buffer = ReplayBuffer(state_dim, action_dim, offline_buffer_size, device)
        self.online_buffer = ReplayBuffer(state_dim, action_dim, online_buffer_size, device)

    def load_offline_dataset(self, data: Dict[str, np.ndarray]):
        self.offline_buffer.load_d4rl_dataset(data)

    def add_online_transition(self, state, action, reward, next_state, done):
        self.online_buffer.add_transition(state, action, reward, next_state, done)

    def sample(self, batch_size: int, online_ratio: float = 0.5) -> List[torch.Tensor]:
        n_online = int(batch_size * online_ratio)
        n_offline = batch_size - n_online

        if self.online_buffer._size < n_online:
            return self.offline_buffer.sample(batch_size)

        offline_batch = self.offline_buffer.sample(n_offline)
        online_batch = self.online_buffer.sample(n_online)

        combined_batch = []
        for i in range(len(offline_batch)):
            combined = torch.cat([offline_batch[i], online_batch[i]], dim=0)
            combined_batch.append(combined)
        return combined_batch


# ------------- Networks (Customizable) -------------
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
        self.attention = SelfAttention()
        self.kinematics = 'holonomic'
        self.multiagent_training = True

    def forward(self, obs: torch.Tensor) -> Normal:
        x = self.attention(obs)
        mean = self.net(x)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cuda"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        # 训练时可以采样，评估时通常用 mean
        action = dist.mean
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
        return self.q1(x), self.q2(x)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


# ------------- AWAC Algorithm -------------
class AWAC:
    def __init__(
            self,
            max_action: float,
            actor: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            q_network: nn.Module,
            q_optimizer: torch.optim.Optimizer,
            beta: float = 1.0,
            discount: float = 0.99,
            tau: float = 0.005,
            device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.actor = actor

        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer

        self.beta = beta
        self.discount = discount
        self.tau = tau
        self.device = device
        self.total_it = 0

    def _update_critic(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_observations: torch.Tensor,
            dones: torch.Tensor,
            log_dict: Dict
    ):
        with torch.no_grad():
            # AWAC: Target Q uses actions sampled from current policy (Off-Policy style)
            # r + gamma * E[Q(s', a')]
            if isinstance(self.actor, DeterministicPolicy):
                next_actions = self.actor(next_observations)
            else:
                dist = self.actor(next_observations)
                next_actions = dist.sample()

            target_q1, target_q2 = self.q_target.both(next_observations, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - dones) * self.discount * target_q

        # Current Q
        current_q1, current_q2 = self.qf.both(observations, actions)

        # Standard MSE Loss (Different from IQL Expectile Loss)
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        log_dict["q_loss"] = q_loss.item()
        log_dict["q_mean"] = current_q1.mean().item()

    def _update_actor(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            log_dict: Dict
    ):
        # AWAC Actor Update: Maximize E[ exp(A/beta) * log_pi(a|s) ]
        with torch.no_grad():
            # 1. Q(s, a_buffer)
            q1, q2 = self.qf.both(observations, actions)
            q_val = torch.min(q1, q2)

            # 2. V(s) approx via E_{a~pi}[Q(s, a)]
            if isinstance(self.actor, DeterministicPolicy):
                curr_actions = self.actor(observations)
            else:
                dist = self.actor(observations)
                curr_actions = dist.sample()

            v1, v2 = self.qf.both(observations, curr_actions)
            v_val = torch.min(v1, v2)

            # 3. Advantage
            adv = q_val - v_val

            # [重要优化] Advantage Normalization: 提升训练稳定性
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # 4. Weights
            weights = torch.exp(adv / self.beta)
            weights = torch.clamp(weights, max=20.0)

        # Actor Loss: Weighted Likelihood
        if isinstance(self.actor, DeterministicPolicy):
            # For deterministic, approximate with weighted MSE
            curr_pi = self.actor(observations)
            actor_loss = torch.mean(weights * torch.sum((curr_pi - actions) ** 2, dim=1))
        else:
            dist = self.actor(observations)
            # log_prob shape: (batch,)
            log_prob = dist.log_prob(actions).sum(-1)
            actor_loss = -torch.mean(weights * log_prob)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        log_dict["actor_loss"] = actor_loss.item()
        log_dict["weights_mean"] = weights.mean().item()
        log_dict["adv_mean"] = adv.mean().item()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1

        # Batch unpacking
        # 兼容 DualBuffer 或 ReplayBuffer 返回的格式 (前5个通常是 s,a,r,ns,d)
        states, actions, rewards, next_states, dones = batch[:5]
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)

        log_dict = {}

        # 1. Critic Update
        self._update_critic(states, actions, rewards, next_states, dones, log_dict)

        # 2. Actor Update
        self._update_actor(states, actions, log_dict)

        # 3. Target Update
        soft_update(self.q_target, self.qf, self.tau)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(self.device)
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict.get("total_it", 0)


# ------------- Utils -------------
def transform(state, device):
    state_tensor = torch.cat([torch.Tensor([state.self_state + human_state]).to(device)
                              for human_state in state.human_states], dim=0)
    state_tensor = rotate(state_tensor)
    return state_tensor


def rotate(state):
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
    new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
    return new_state


def set_seed(seed: int, env: Optional[gym.Env] = None):
    if env is not None:
        env.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# ------------- Eval & Test Functions -------------
@torch.no_grad()
def eval_actor(
        env: gym.Env, actor: nn.Module, list1: List, list2: List, list3: List, list4: List, device: str,
        n_episodes: int, seed: int, logger: Optional[logging.Logger] = None, global_step: Optional[int] = None):
    # 这里的 env 配置逻辑保留你原来的写法
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str,
                        default='D:\Program Files (x86)\IDE\JetBrains\PycharmProjects\CrowdNav-master_danren\crowd_nav\configs\env.config')
    args, _ = parser.parse_known_args()

    actor.eval()
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)

    # 注意：这里可能会重新创建 env，为了避免冲突最好直接传进来的 env
    # 但为了保持一致性，照抄你的逻辑
    eval_env = gym.make('CrowdSim-v0')
    eval_env.configure(env_config)
    robot = Robot(env_config, 'robot')
    eval_env.set_robot(robot)
    robot.set_policy(actor)

    success = 0
    collision = 0
    run_time = 0
    avg_nav_time = 0
    avg_reward = 0

    # 使用随机种子偏移
    eval_env.seed(seed + 100)

    for i in range(n_episodes):
        obs = eval_env.reset(phase='train', test_case=i)
        done = False
        rewards = 0
        gamma = 1.0

        while not done:
            joint_state = JointState(robot.get_full_state(), obs)
            state = to_np(transform(joint_state, device).view(1, -1).squeeze(0))
            action = actor.act(state, device=device)
            action_obj = ActionXY(action[0], action[1])
            obs, reward, done, info = eval_env.step(action_obj)
            rewards += reward * gamma
            gamma *= 0.99

        if isinstance(info, ReachGoal):
            success += 1
            avg_nav_time += eval_env.global_time
        elif isinstance(info, Collision):
            collision += 1
        elif isinstance(info, Timeout):
            run_time += 1
        avg_reward += rewards

    if success != 0:
        avg_nav_time = avg_nav_time / success

    avg_reward /= n_episodes
    success_rate = success / n_episodes

    if logger is not None:
        logger.info(
            "[EVAL] step=%s | succ=%.3f | coll=%.3f | reward=%.4f",
            str(global_step), success_rate, collision / n_episodes, avg_reward
        )

    list1.append(success_rate)
    list2.append(collision / n_episodes)
    list3.append(avg_nav_time)
    list4.append(avg_reward)

    actor.train()
    return {"success_rate": success_rate}


@torch.no_grad()
def test_actor(
        env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int,
        logger: Optional[logging.Logger] = None):
    # 配置 Robot
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_config', type=str,
                        default="D:\Program Files (x86)\IDE\JetBrains\PycharmProjects\CrowdNav-master_danren\crowd_nav\configs\env.config")
    args, _ = parser.parse_known_args()

    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)

    test_env = gym.make('CrowdSim-v0')
    test_env.configure(env_config)
    robot = Robot(env_config, 'robot')
    test_env.set_robot(robot)
    robot.set_policy(actor)

    actor.eval()

    success = 0
    collision = 0
    run_time = 0
    avg_reward = 0
    nav_time = 0
    for _ in range(n_episodes):
        obs = test_env.reset(phase='test')
        done = False
        rewards = 0
        gamma = 1.0

        while not done:
            joint_state = JointState(robot.get_full_state(), obs)
            state = to_np(transform(joint_state, device).view(1, -1).squeeze(0))
            action = actor.act(state, device=device)
            obs, reward, done, info = test_env.step(ActionXY(action[0], action[1]))
            rewards += reward * gamma
            gamma *= 0.99

        if isinstance(info, ReachGoal):
            success += 1
            nav_time += test_env.global_time
        elif isinstance(info, Collision):
            collision += 1
        elif isinstance(info, Timeout):
            run_time += 1
        avg_reward += rewards
        test_env.render(mode='traj')
        plt.show()
    success_rate = success / n_episodes
    avg_nav_time = nav_time / success
    if logger:
        logger.info(f"[TEST] Success Rate: {success_rate:.3f}, Collision Rate: {collision / n_episodes:.3f}, Navtime: {avg_nav_time:.3f}, Avg Reward: {avg_reward / n_episodes:.3f}")

    actor.train()


# ------------- Main Train Loop -------------
@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)

    state_dim = 65
    action_dim = 2
    max_action = float(1)

    # 1. Load Data
    file_path = "D:\\Program Files (x86)\\IDE\\JetBrains\\PycharmProjects\\CrowdNav-master_danren\\crowd_nav\\Crowd_nav_5\\Crowd_nav_5.hdf5"
    with h5py.File(file_path, 'r') as f:
        dataset = {k: np.array(v) for k, v in f.items()}

    # 2. Setup Buffer
    dual_buffer = DualReplayBuffer(
        state_dim, action_dim, config.buffer_size, config.online_buffer_max_size, config.device
    )
    dual_buffer.load_offline_dataset(dataset)

    # 3. Setup Networks
    q_network = TwinQ(state_dim, action_dim).to(config.device)

    if config.iql_deterministic:
        actor = DeterministicPolicy(state_dim, action_dim, max_action, dropout=config.actor_dropout)
    else:
        # AWAC 推荐使用 GaussianPolicy
        actor = GaussianPolicy(state_dim, action_dim, max_action, dropout=config.actor_dropout)
    actor = actor.to(config.device)

    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    # 4. Setup Agent (AWAC)
    trainer = AWAC(
        max_action=max_action,
        actor=actor,
        actor_optimizer=actor_optimizer,
        q_network=q_network,
        q_optimizer=q_optimizer,
        beta=config.beta,
        discount=config.discount,
        tau=config.tau,
        device=config.device
    )

    # 5. Logging
    os.makedirs(config.checkpoints_path, exist_ok=True)
    logger = setup_logger(config.checkpoints_path, name="awac")
    logger.info("Start training AWAC | Env: %s | Beta: %.2f", config.env, config.beta)

    set_seed(config.seed, env)

    list1, list2, list3, list4 = [], [], [], []

    # 6. Offline Training Loop
    logger.info("========== STAGE 1: OFFLINE PRE-TRAINING ==========")
    for t in range(int(config.max_timesteps)):
        # 只从离线 Buffer 采样
        batch = dual_buffer.offline_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]

        log_dict = trainer.train(batch)

        if (t + 1) % config.eval_freq == 0:
            logger.info(f"Step {t + 1} | Q_Loss: {log_dict['q_loss']:.4f} | Actor_Loss: {log_dict['actor_loss']:.4f}")
            eval_actor(env, actor, list1, list2, list3, list4,
                       device=config.device, n_episodes=config.n_episodes, seed=config.seed, logger=logger,
                       global_step=t + 1)

    # 7. Save Checkpoints
    logger.info("Offline training finished. Saving checkpoints...")

    # Save AWAC Full State
    torch.save(trainer.state_dict(), os.path.join(config.checkpoints_path, "awac_offline.pt"))
    # Save Actor Only
    torch.save(actor.state_dict(), os.path.join(config.checkpoints_path, "actor_offline.pt"))

    logger.info("Saved models to %s", config.checkpoints_path)


if __name__ == "__main__":
    train()