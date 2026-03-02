import os
import sys
import logging
import h5py
import numpy as np
import torch
import gym
import pyrallis

from crowd_nav.iql_rrnd import (
    TrainConfig,
    ReplayBuffer,
    DualReplayBuffer,
    TwinQ,
    ValueFunction,
    GaussianPolicy,
    DeterministicPolicy,
    ImplicitQLearning,
    set_seed,
    eval_actor,
    test_actor,
    online_finetune_weighted_loss,
)

from crowd_nav.policy.rnd import RND


def setup_logger(log_path: str) -> logging.Logger:
    """
    和你离线训练时尽量保持一致的 logger：
    - 输出到控制台
    - 同时写到 train_eval.log（这里用追加模式）
    """
    logger = logging.getLogger("online_finetune")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 防止重复日志

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    # 文件日志（追加写）
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
    """
    只做在线微调：
      1. 从离线 checkpoint 恢复 IQL + RND
      2. 用离线 hdf5 构造 replay buffer
      3. 调 online_finetune_with_reverse_rnd 做在线微调
      4. 前后各 test 一次，并保存在线后的权重
    """

    # ===== 一些基本设置 =====
    os.makedirs(config.checkpoints_path, exist_ok=True)
    log_file = os.path.join(config.checkpoints_path, "train_eval_iql.log")  # 和原来一致
    logger = setup_logger(log_file)

    logger.info("========== ONLINE FINETUNE ONLY ==========")
    logger.info("Using device: %s", config.device)

    # ===== 构造环境（用于设随机种子 & test）=====
    env = gym.make(config.env)
    set_seed(config.seed, env)


    # ===== 载入离线 HDF5 数据集，构建 ReplayBuffer =====
    # ⚠️ 这里路径请改成你自己实际的 hdf5 路径
    file_path = (
        "D:\\Program Files (x86)\\IDE\\JetBrains\\PycharmProjects\\CrowdNav-master_danren\\crowd_nav\\Crowd_nav_5\\Crowd_nav_5.hdf5"
    )
    logger.info("Loading offline dataset from: %s", file_path)
    file_path_online = (
        "D:\\Program Files (x86)\\IDE\\JetBrains\\PycharmProjects\\CrowdNav-master_danren\\crowd_nav\\Crowd_nav_5_iql_online\\Crowd_nav_5_iql_online.hdf5"
    )

    with h5py.File(file_path, "r") as f:
        dataset = {k: np.array(v) for k, v in f.items()}

    logger.info(
        "Offline dataset loaded. states shape=%s, actions shape=%s, rewards in [%.3f, %.3f]",
        dataset["observations"].shape,
        dataset["actions"].shape,
        dataset["rewards"].min(),
        dataset["rewards"].max(),
    )
    # with h5py.File(file_path_online, "r") as f:
    #     dataset_online = {k: np.array(v) for k, v in f.items()}
    #
    # logger.info(
    #     "Offline dataset loaded. states shape=%s, actions shape=%s, rewards in [%.3f, %.3f]",
    #     dataset_online["observations"].shape,
    #     dataset_online["actions"].shape,
    #     dataset_online["rewards"].min(),
    #     dataset_online["rewards"].max(),
    # )


    state_dim = 65
    action_dim = 2
    max_action = float(1)


    # ===== 构建 RND，并从离线 checkpoint 恢复 =====
    rnd = RND(
        input_dim=state_dim,
        hidden_dim=config.rnd_hidden_dim,
        output_dim= 128,
        lr=1e-4,
        device=config.device,
    )

    rnd_offline_path = os.path.join("D:\Program Files (x86)\IDE\JetBrains\PycharmProjects\CrowdNav-master_danren\crowd_nav\data\output_iql", "rnd_offline.pt")
    logger.info("Loading offline RND checkpoint from: %s", rnd_offline_path)
    rnd_ckpt = torch.load(rnd_offline_path, map_location=config.device)
    rnd.load_state_dict(rnd_ckpt["rnd_state"])
    rnd.optimizer.load_state_dict(rnd_ckpt["rnd_optimizer"])
    logger.info("Offline RND checkpoint loaded.")

    # replay_buffer = ReplayBuffer(
    #     state_dim=state_dim,
    #     action_dim=action_dim,
    #     buffer_size=config.buffer_size,
    #     device=config.device,
    # )

    # [修改] 初始化双缓冲区
    dual_replay_buffer = DualReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        offline_buffer_size=config.buffer_size,  # 比如 200W
        online_buffer_size=config.online_buffer_max_size,  # 比如 10W
        device=config.device,
    )

    # 1. 计算离线数据的 RND Error
    logger.info("Calculating offline RND statistics...")
    all_offline_errors = rnd.compute_dataset_novelty(dataset["observations"], batch_size=1024)

    # 2. 获取统计量 (关键步骤！)
    rnd_min = all_offline_errors.min().item()
    rnd_max = all_offline_errors.max().item()
    rnd_range = rnd_max - rnd_min + 1e-8  # 避免除以0

    logger.info(f"Offline RND Stats -> Min: {rnd_min:.6f}, Max: {rnd_max:.6f}")

    # 3. 归一化离线权重并存入 Buffer
    norm_errors = (all_offline_errors - rnd_min) / rnd_range
    k = 0  # 调节系数
    weights = 1.0 / (1.0 + k * norm_errors)

    # 存入 Buffer (假设你修改了 load_d4rl_dataset 允许直接传 weights)
    dual_replay_buffer.load_offline_dataset(dataset, weights=weights)
    logger.info("Replay buffer loaded with %d transitions.", dual_replay_buffer.offline_buffer._size)

    # 4. 把统计量存入 config 或者传给 trainer
    # replay_buffer.rnd_min = rnd_min
    # replay_buffer.rnd_max = rnd_max





    # ===== 构建 IQL 网络 & Optimizer，然后从离线 checkpoint 恢复 =====
    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)

    if config.iql_deterministic:
        actor = DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    else:
        actor = GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    actor = actor.to(config.device)

    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    trainer = ImplicitQLearning(
        q_network=q_network,
        v_network=v_network,
        actor=actor,
        v_optimizer=v_optimizer,
        q_optimizer=q_optimizer,
        actor_optimizer=actor_optimizer,
        tau=config.tau,
        discount=config.discount,
        beta=config.beta,
        iql_tau=config.iql_tau,
        max_steps=None,
        device=config.device,
        max_action=max_action,
    )

    # === 从离线阶段保存的 iql_offline.pt 中恢复 trainer ===
    offline_iql_path = os.path.join(config.checkpoints_path, "iql_offline.pt")
    logger.info("Loading offline IQL checkpoint from: %s", offline_iql_path)
    iql_state = torch.load(offline_iql_path, map_location=config.device)
    trainer.load_state_dict(iql_state)
    logger.info("Offline IQL checkpoint loaded.")



    # ===== 在线微调前，先测一遍当前策略表现 =====
    logger.info("========== BEFORE ONLINE FINETUNE ==========")
    test_actor(
        env,
        trainer.actor,
        device=config.device,
        n_episodes=config.n_episodes,
        seed=config.seed,
        logger=logger,
    )

    # ===== 正式开始在线微调（反向 RND 加权采样）=====
    logger.info("========== START ONLINE FINETUNE ==========")
    online_finetune_weighted_loss(
        trainer=trainer,
        rnd=rnd,
        dual_buffer=dual_replay_buffer, # 传入双缓冲区
        config=config,
        logger=logger,
    )
    # online_finetune_without_reverse_rnd(
    #     trainer=trainer,
    #     replay_buffer=replay_buffer,
    #     config=config,
    #     logger=logger,
    # )

    # ===== 在线微调后，再测试一次 =====
    logger.info("========== AFTER ONLINE FINETUNE ==========")
    test_actor(
        env,
        trainer.actor,
        device=config.device,
        n_episodes=config.n_episodes,
        seed=config.seed,
        logger=logger,
    )

    # ===== 保存在线微调后的权重 =====
    iql_online_path = os.path.join(
        config.checkpoints_path, "iql_rnd_online.pt"
    )
    actor_online_path = os.path.join(config.checkpoints_path, "actor_online.pt")
    rnd_online_path = os.path.join(config.checkpoints_path, "rnd_online.pt")

    torch.save(trainer.state_dict(), iql_online_path)
    torch.save(trainer.actor.state_dict(), actor_online_path)
    torch.save(
        {
            "rnd_state": rnd.state_dict(),
            "rnd_optimizer": rnd.optimizer.state_dict(),
        },
        rnd_online_path,
    )

    logger.info("Saved online-finetuned IQL to   %s", iql_online_path)
    logger.info("Saved online-finetuned actor to %s", actor_online_path)
    logger.info("Saved online-finetuned RND to   %s", rnd_online_path)
    logger.info("========== ONLINE FINETUNE DONE ==========")


if __name__ == "__main__":
    main()
