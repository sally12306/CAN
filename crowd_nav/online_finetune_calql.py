import os
import sys
import logging
import h5py
import numpy as np
import torch
import gym
import pyrallis
import configparser
from copy import deepcopy

# ==========================================
# 1. 导入修复：显式导入 Scalar 防止 torch.load 报错
# ==========================================
from train_calql import (
    TrainConfig,
    CalQL,
    ReplayBuffer,
    FullyConnectedQFunction,
    TanhGaussianPolicy,
    Scalar,  # <--- 必须显式导入
    ReparameterizedTanhGaussian,  # <--- 必须显式导入
    set_seed,
    test_actor,
    transform,
    to_np,
    ActionXY,
    JointState,
    ReachGoal,
    Collision,
    Timeout,
    get_return_to_go
)

from crowd_sim.envs.utils.robot import Robot


# ==========================================
# 2. Bug 修复：Monkey Patch 覆盖 ReplayBuffer.sample
# ==========================================
def correct_sample(self, batch_size: int):
    """
    修复原始代码中 min(size, pointer) 导致的 crash 和逻辑错误。
    正确做法是在 [0, size) 范围内采样。
    """
    # 确保 high >= 1，防止 batch_size_online 计算异常导致 0
    high = self._size
    if high <= 0:
        # 如果 buffer 是空的，这应该被外层逻辑拦截，但为了安全：
        return None

        # 正确的采样逻辑：在有效大小内随机采样
    indices = np.random.randint(0, high, size=batch_size)

    states = self._states[indices]
    actions = self._actions[indices]
    rewards = self._rewards[indices]
    next_states = self._next_states[indices]
    dones = self._dones[indices]
    mc_returns = self._mc_returns[indices]
    return [states, actions, rewards, next_states, dones, mc_returns]


# 应用修复：将 ReplayBuffer 类的 sample 方法替换为上面的正确版本
ReplayBuffer.sample = correct_sample


# ==========================================
# 3. 日志与工具函数
# ==========================================
def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("calql_online_finetune")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def compute_mc_returns(rewards, dones, discount=0.99):
    """手动计算轨迹的 Return-to-Go (MC Returns)"""
    returns = []
    G = 0
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            G = 0
        G = r + discount * G
        returns.insert(0, G)
    return returns


@pyrallis.wrap()
def main(config: TrainConfig):
    # ================= 配置调整 =================
    finetune_lr = 1e-5  # 微调学习率
    online_alpha = 0.5  # 在线阶段保守系数 (0.5 ~ 1.0)
    mixing_ratio = 0.5  # 混合比例

    os.makedirs(config.checkpoints_path, exist_ok=True)
    logger = setup_logger(os.path.join(config.checkpoints_path, "train_eval_calql_online.log"))

    logger.info("========== Cal-QL ONLINE FINETUNE (Patched) ==========")
    logger.info(f"Device: {config.device} | Alpha: {online_alpha} | LR: {finetune_lr}")

    # ================= 环境初始化 =================
    env = gym.make(config.env)
    set_seed(config.seed, env)

    # 请确认以下路径正确
    env_config_path = "D:\\Program Files (x86)\\IDE\\JetBrains\\PycharmProjects\\CrowdNav-master_danren\\crowd_nav\\configs\\env.config"
    env_config_file = configparser.RawConfigParser()
    env_config_file.read(env_config_path)
    env.configure(env_config_file)

    robot = Robot(env_config_file, 'robot')
    env.set_robot(robot)

    state_dim = 65
    action_dim = 2
    max_action = 1.0

    # ================= 模型初始化 =================
    critic_1 = FullyConnectedQFunction(state_dim, action_dim, config.orthogonal_init, config.q_n_hidden_layers).to(
        config.device)
    critic_2 = FullyConnectedQFunction(state_dim, action_dim, config.orthogonal_init, config.q_n_hidden_layers).to(
        config.device)
    actor = TanhGaussianPolicy(state_dim, action_dim, max_action, orthogonal_init=config.orthogonal_init).to(
        config.device)

    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=finetune_lr)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=finetune_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=finetune_lr)

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
        "target_entropy": -2.0,
        "alpha_multiplier": config.alpha_multiplier,
        "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
        "backup_entropy": config.backup_entropy,
        "policy_lr": finetune_lr,
        "qf_lr": finetune_lr,
        "cql_alpha": online_alpha,
        "cql_n_actions": config.cql_n_actions,
        "cql_importance_sample": config.cql_importance_sample,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "cql_temp": config.cql_temp,
        "cql_max_target_backup": config.cql_max_target_backup,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
    }

    trainer = CalQL(**kwargs)

    # ================= 加载离线模型 =================
    offline_model_path = config.load_model

    if os.path.exists(offline_model_path) and offline_model_path != "":
        logger.info(f"Loading offline model: {offline_model_path}")
        # 这里的 map_location 和 import 修复保证了加载成功
        state_dict = torch.load(offline_model_path, map_location=config.device)
        trainer.load_state_dict(state_dict)

        # 重置优化器学习率
        for opt in [trainer.actor_optimizer, trainer.critic_1_optimizer, trainer.critic_2_optimizer]:
            for param_group in opt.param_groups:
                param_group['lr'] = finetune_lr
    else:
        logger.warning(f"No offline model loaded! Path: {offline_model_path}")

    # 开启校准模式
    trainer._calibration_enabled = True
    trainer.cql_alpha = online_alpha

    # ================= 数据集加载 =================
    # 离线 Buffer
    offline_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)

    hdf5_path = "D:\\Program Files (x86)\\IDE\\JetBrains\\PycharmProjects\\CrowdNav-master_danren\\crowd_nav\\Crowd_nav_5\\Crowd_nav_5.hdf5"
    logger.info(f"Loading offline dataset: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        dataset = {k: np.array(v) for k, v in f.items()}

    mc_returns = get_return_to_go(dataset, env, config)
    dataset["mc_returns"] = np.array(mc_returns)

    offline_buffer.load_d4rl_dataset(dataset)
    logger.info(f"Offline Buffer Size: {offline_buffer._size}")

    # 在线 Buffer
    online_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)

    # ================= 在线微调循环 =================
    batch_size_offline = int(config.batch_size * mixing_ratio)
    batch_size_online = config.batch_size - batch_size_offline

    online_episodes = 50000
    min_steps_before_train = 1000
    total_online_steps = 0

    logger.info("Starting Online Finetuning Loop...")

    test_actor(env, trainer.actor, config.device, n_episodes=config.n_episodes, seed=config.seed)

    for ep in range(online_episodes):
        obs = env.reset(phase='train')
        done = False

        traj_states, traj_actions, traj_rewards = [], [], []
        traj_next_states, traj_dones = [], []

        ep_reward = 0
        ep_steps = 0

        # 统计量
        success = 0
        collision = 0
        timeout = 0
        episode_reward = 0.0
        nav_time = 0.0
        info = None

        # === 收集轨迹 ===
        while not done:
            joint_state = JointState(robot.get_full_state(), obs)
            state = to_np(transform(joint_state, config.device).view(1, -1).squeeze(0))

            trainer.actor.train()  # 开启探索模式

            with torch.no_grad():
                raw_action = trainer.actor.act(state, device=config.device)

            action = ActionXY(raw_action[0], raw_action[1])
            next_obs, reward, done, info = env.step(action)

            next_joint_state = JointState(robot.get_full_state(), next_obs)
            next_state = to_np(transform(next_joint_state, config.device).view(1, -1).squeeze(0))

            traj_states.append(state)
            traj_actions.append(raw_action)
            traj_rewards.append(reward)
            traj_next_states.append(next_state)
            traj_dones.append(float(done))

            obs = next_obs
            ep_reward += reward
            ep_steps += 1
            total_online_steps += 1

        # === 处理轨迹并存入 Buffer (计算 MC Return) ===
        traj_mc_returns = compute_mc_returns(traj_rewards, traj_dones, config.discount)

        for i in range(len(traj_states)):
            online_buffer.add_transition(
                traj_states[i], traj_actions[i], traj_rewards[i],
                traj_next_states[i], traj_dones[i]
            )
            # 手动注入 MC Return
            idx = (online_buffer._pointer - 1) % online_buffer._buffer_size
            online_buffer._mc_returns[idx] = traj_mc_returns[i]


        # Episode 结束统计
        if isinstance(info, ReachGoal):
            success = 1
            nav_time = env.global_time
        elif isinstance(info, Collision):
            collision = 1
        elif isinstance(info, Timeout):
            timeout = 1
        episode_reward = ep_reward



        # === 训练更新 ===
        if online_buffer._size >= batch_size_online and total_online_steps >= min_steps_before_train:
            train_steps = 1

            for _ in range(train_steps):
                off_batch = offline_buffer.sample(batch_size_offline)
                off_batch = [b.to(config.device) for b in off_batch]

                # 这里的 sample 已经被 patch 过了，不会再报错
                on_batch = online_buffer.sample(batch_size_online)
                on_batch = [b.to(config.device) for b in on_batch]

                batch = [torch.cat([o, n], dim=0) for o, n in zip(off_batch, on_batch)]

                trainer.train(batch)

        # === 日志与评估 ===
        # success = isinstance(info, ReachGoal)
        # if (ep + 1) % 10 == 0:
        #     logger.info(f"Ep {ep + 1} | Rew: {ep_reward:.2f} | Succ: {int(success)} | Steps: {ep_steps}")

        if logger is not None:
            logger.info(
                "[ONLINE_CALQL] ep=%d/%d | succ=%d | coll=%d | timeout=%d | nav_time=%.4f | reward=%.4f | buffer_size=%d",
                ep + 1,
                online_episodes,
                success,
                collision,
                timeout,
                nav_time,
                episode_reward,
                online_buffer._size,
            )


        if (ep + 1) % config.eval_freq == 0:
            logger.info(f"Evaluating at episode {ep + 1}...")
            test_actor(env, trainer.actor, config.device, n_episodes=config.n_episodes, seed=config.seed)
            # save_path = os.path.join(config.checkpoints_path, f"calql_online_{ep + 1}.pt")
            # torch.save(trainer.state_dict(), save_path)

    final_path = os.path.join(config.checkpoints_path, "calql_online_final.pt")
    torch.save(trainer.state_dict(), final_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()