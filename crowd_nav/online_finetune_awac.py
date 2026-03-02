import os
import sys
import logging
import h5py
import numpy as np
import torch
import gym
import pyrallis
import configparser

# ============================================================
# 关键：直接复用 train_awac.py 中的网络、配置和工具
# 确保你的 train_awac.py 文件就在 crowd_nav 目录下
# ============================================================
from crowd_nav.train_awac import (
    TrainConfig,
    DualReplayBuffer,
    TwinQ,
    GaussianPolicy,
    DeterministicPolicy,
    AWAC,
    set_seed,
    test_actor,
    transform,
    to_np,
)

# 引入环境相关的工具类
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.info import ReachGoal, Collision, Timeout


def setup_logger(log_path: str) -> logging.Logger:
    """
    配置日志：同时输出到控制台和文件
    """
    logger = logging.getLogger("awac_online_finetune")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 清除已有的 handler 防止重复打印
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    # 文件日志 (mode='w' 覆盖写，方便每次看新的)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 控制台日志
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


@pyrallis.wrap()
def main(config: TrainConfig):
    # ================= 1. 初始化与配置调整 =================
    # 在线微调建议使用较小的 beta 以保持稳定 (0.5 ~ 1.0)
    # 如果离线训练时用了归一化，这里保持一致
    # config.beta = 1.0

    # 微调阶段学习率建议降低，防止破坏预训练权重
    finetune_lr = 1e-5

    os.makedirs(config.checkpoints_path, exist_ok=True)
    logger = setup_logger(os.path.join(config.checkpoints_path, "train_eval_awac_online.log"))

    logger.info("========== AWAC ONLINE FINETUNE (REUSING train_awac.py) ==========")
    logger.info(f"Device: {config.device} | Beta: {config.beta} | LR: {finetune_lr}")

    # 环境设置
    env = gym.make(config.env)
    set_seed(config.seed, env)

    # 读取环境配置 (用于 Robot 和后续 env.configure)
    # 请确保此路径指向正确的 env.config 文件
    env_config_file = "D:\\Program Files (x86)\\IDE\\JetBrains\\PycharmProjects\\CrowdNav-master_danren\\crowd_nav\\configs\\env.config"
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env.configure(env_config)  # 配置环境参数

    state_dim = 65
    action_dim = 2
    max_action = 1.0

    # ================= 2. 复用 train_awac.py 中的网络 =================
    # 初始化 Q 网络
    q_network = TwinQ(state_dim, action_dim).to(config.device)

    # 初始化 Actor (使用和离线训练一致的策略类型)
    if config.iql_deterministic:
        actor = DeterministicPolicy(state_dim, action_dim, max_action, dropout=config.actor_dropout)
    else:
        actor = GaussianPolicy(state_dim, action_dim, max_action, dropout=config.actor_dropout)
    actor = actor.to(config.device)

    # 优化器 (使用微调的学习率)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=finetune_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=finetune_lr)

    # 初始化 AWAC 算法实例
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

    # ================= 3. 加载离线预训练权重 =================
    offline_ckpt_path = os.path.join(config.checkpoints_path, "awac_online_final.pt")

    logger.info(f"Loading offline checkpoint from: {offline_ckpt_path}")
    if os.path.exists(offline_ckpt_path):
        checkpoint = torch.load(offline_ckpt_path, map_location=config.device)
        trainer.load_state_dict(checkpoint)
        # 注意：如果 checkpoint 里包含旧的 optimizer 状态（大 LR），
        # 加载后建议手动覆盖 LR，或者重新初始化 optimizer (这里我们选择重新初始化 optimizer，只加载网络权重)
        # 如果你想完全恢复状态：trainer.load_state_dict(checkpoint) 会覆盖 optimizer。
        # 为了微调稳定性，我们手动把 optimizer 的 LR 设回去：
        for param_group in trainer.q_optimizer.param_groups:
            param_group['lr'] = finetune_lr
        for param_group in trainer.actor_optimizer.param_groups:
            param_group['lr'] = finetune_lr

        logger.info(">>> Checkpoint loaded. Optimizer LR reset to %.1e", finetune_lr)
    else:
        raise FileNotFoundError(f"未找到离线模型文件: {offline_ckpt_path}")

    # ================= 4. 准备 Buffer (用于混合采样) =================
    dual_buffer = DualReplayBuffer(
        state_dim, action_dim, config.buffer_size, config.online_buffer_max_size, config.device
    )

    hdf5_path = "D:\\Program Files (x86)\\IDE\\JetBrains\\PycharmProjects\\CrowdNav-master_danren\\crowd_nav\\Crowd_nav_5\\Crowd_nav_5.hdf5"
    logger.info(f"Loading offline dataset: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        dataset = {k: np.array(v) for k, v in f.items()}

    dual_buffer.load_offline_dataset(dataset)
    logger.info(f"Offline Data Size: {dual_buffer.offline_buffer._size}")

    # ================= 5. 设置机器人与策略 =================
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    robot.set_policy(trainer.actor)

    # ================= 6. 在线微调循环 =================
    logger.info("========== START ONLINE FINETUNE ==========")

    test_actor(env, trainer.actor, config.device, n_episodes=config.n_episodes, seed=config.seed, logger=logger)

    online_episodes = 50000
    # 安全阈值：建议攒够 batch_size * 2 的数据再开始混合训练
    min_safe_online = max(512, 1000)

    for ep in range(online_episodes):
        obs = env.reset(phase='train')
        done = False
        ep_reward = 0
        ep_steps = 0



        # 统计量
        success = 0
        collision = 0
        timeout = 0
        episode_reward = 0.0
        nav_time = 0.0
        info = None

        # 噪声衰减：前 20% 阶段保持较高噪声，后面衰减到小噪声
        if ep < online_episodes * 0.2:
            noise_scale = 0.2
        else:
            noise_scale = 0.05

        while not done:
            # 1. 状态处理
            joint_state = JointState(robot.get_full_state(), obs)
            state_vec = to_np(transform(joint_state, config.device).view(1, -1).squeeze(0))

            # 2. 动作选择
            raw_action = trainer.actor.act(state_vec, device=config.device)

            # 3. 添加探索噪声 (使用 np.clip 替代 clamp_xy)
            noise = np.random.normal(0, noise_scale, size=2)
            action_np = np.clip(raw_action + noise, -1, 1)

            vx, vy = action_np[0], action_np[1]

            # 4. 环境交互
            next_obs, reward, done, info = env.step(ActionXY(vx, vy))
            ep_reward += reward
            ep_steps += 1

            next_joint_state = JointState(robot.get_full_state(), next_obs)
            next_state_vec = to_np(transform(next_joint_state, config.device).view(1, -1).squeeze(0))

            # 5. 存入在线 Buffer
            dual_buffer.add_online_transition(
                state_vec, action_np, reward, next_state_vec, float(done)
            )

            obs = next_obs


        # Episode 结束统计
        if isinstance(info, ReachGoal):
            success = 1
            nav_time = env.global_time
        elif isinstance(info, Collision):
            collision = 1
        elif isinstance(info, Timeout):
            timeout = 1
        episode_reward = ep_reward


        # 6. 训练 (混合采样)
        if dual_buffer.online_buffer._size >= min_safe_online:
            # 混合比例：建议 80% 离线 + 20% 在线 (online_ratio=0.2) 以保持分布稳定
            # 或者 50% + 50%
            batch = dual_buffer.sample(config.batch_size, online_ratio=0.5)
            batch = [b.to(config.device) for b in batch]

            log_dict = trainer.train(batch)



        if logger is not None:
            logger.info(
                "[ONLINE_AWAC] ep=%d/%d | succ=%d | coll=%d | timeout=%d | nav_time=%.4f | reward=%.4f | buffer_size=%d",
                ep + 1,
                online_episodes,
                success,
                collision,
                timeout,
                nav_time,
                episode_reward,
                dual_buffer.online_buffer._size,
            )



        # 日志记录
        if (ep + 1) % 10 == 0:
            success = isinstance(info, ReachGoal)
            collision = isinstance(info, Collision)

            q_loss = log_dict.get('q_loss', 0.0) if 'log_dict' in locals() else 0.0
            actor_loss = log_dict.get('actor_loss', 0.0) if 'log_dict' in locals() else 0.0

            # logger.info(
            #     f"Ep {ep + 1} | Rew: {ep_reward:.2f} | Succ: {int(success)} | Coll: {int(collision)} | "
            #     f"Loss: Q={q_loss:.3f} A={actor_loss:.3f} | Noise: {noise_scale:.2f}"
            # )

        # 定期评估与保存
        if (ep + 1) % config.eval_freq == 0:
            logger.info(f"Evaluating at episode {ep + 1}...")
            # 注意：test_actor 内部会创建新的 env，不影响主循环
            test_actor(env, trainer.actor, config.device, n_episodes=config.n_episodes, seed=config.seed, logger=logger)

            # save_path = os.path.join(config.checkpoints_path, f"awac_online_{ep + 1}.pt")
            # torch.save(trainer.state_dict(), save_path)

    # 最终保存
    final_path = os.path.join(config.checkpoints_path, "awac_online_final.pt")
    torch.save(trainer.state_dict(), final_path)
    logger.info(f"Finished. Model saved to {final_path}")


if __name__ == "__main__":
    main()