import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class RunningMeanStd:
    """动态维护流式数据的均值和方差 (参考 OpenAI Baselines / NGU)"""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class RND(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, lr, device):
        super(RND, self).__init__()
        self.device = device

        # ----------  目标网络（Target Network，固定不训练） ----------
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)

        # ----------  预测网络（Predictor Network，可训练） ----------
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)

        # 固定 target 网络的参数
        for param in self.target.parameters():
            param.requires_grad = False
        # 优化器
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        # [关键改进] 引入运行统计量，用于自适应归一化
        self.rms = RunningMeanStd()



    # # ----------  计算单个状态的新颖度 ----------
    # def get_intrinsic_reward(self, state):
    #     """
    #     计算当前状态的 intrinsic reward（即新颖度）
    #     输入:
    #         state: torch.Tensor, shape=(state_dim,) 或 (1, state_dim)
    #     输出:
    #         novelty: float
    #     """
    #     with torch.no_grad():
    #         state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    #         batch_size = state.shape[0]
    #         state = state.view(batch_size, -1)
    #         target_feat = self.target(state)
    #         pred_feat = self.predictor(state)
    #         # L2 范数作为新颖度指标
    #         novelty = F.mse_loss(pred_feat, target_feat, reduction='none').mean(dim=1)
    #     return novelty.item()

    def get_intrinsic_reward(self, state, update_stats=True):
        """
        计算自适应新颖度 (Adaptive Novelty)
        update_stats: 训练时设为 True，测试时设为 False
        """
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).to(self.device)
            else:
                state_tensor = state

            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)

            target_feat = self.target(state_tensor)
            pred_feat = self.predictor(state_tensor)

            # 1. 计算原始误差 (Batch size, )
            raw_error = F.mse_loss(pred_feat, target_feat, reduction='none').mean(dim=1).cpu().numpy()

        # 2. [自适应核心] 更新统计量
        if update_stats:
            self.rms.update(raw_error)

        # 3. [自适应核心] 标准化：(x - mean) / std
        # 这样无论训练多久，Norm Error > 0 代表比平均水平更陌生，< 0 代表更熟悉
        norm_error = (raw_error - self.rms.mean) / (np.sqrt(self.rms.var) + 1e-8)

        # 4. 截断防止极端值
        return np.clip(norm_error, -5, 5).item()

    def get_raw_error_batch(self, states):
        """
        只计算 Error，不更新 RMS 统计量，用于训练时的 Batch 加权
        """
        self.predictor.eval()  # 临时切到 eval 模式防止 BatchNorm 变动(如果有)
        self.target.eval()

        target_feat = self.target(states)
        pred_feat = self.predictor(states)

        # 计算每个样本的 error (Batch, )
        raw_error = F.mse_loss(pred_feat, target_feat, reduction='none').mean(dim=1)

        self.predictor.train()  # 切回 train 模式
        return raw_error




    # ----------  批量训练 RND ----------
    def train_predictor(self, state_batch):
        """
        输入:
            state_batch: torch.Tensor, shape=(batch_size, state_dim)
        输出:
            loss: float
        """
        batch_size = state_batch.shape[0]
        state_batch = state_batch.view(batch_size, -1)
        target_feat = self.target(state_batch)
        pred_feat = self.predictor(state_batch)

        loss = F.mse_loss(pred_feat, target_feat)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_adaptive(self, online_states, offline_states):
        """
        [防止遗忘] 混合训练接口
        同时输入在线和离线数据，防止 RND 忘记离线分布
        """
        mixed_states = torch.cat([online_states, offline_states], dim=0)
        target_feat = self.target(mixed_states)
        pred_feat = self.predictor(mixed_states)
        loss = F.mse_loss(pred_feat, target_feat)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # ----------  新增：为整个数据集批量计算新颖度 ----------
    def compute_dataset_novelty(self, observations, batch_size=256):
        """
        分批计算大规模数据的 RND error (novelty)
        输入:
            observations: np.ndarray or torch.Tensor, shape=(N, state_dim)
            batch_size: int, 每次推断的大小
        输出:
            novelties: torch.Tensor, shape=(N, 1)
        """
        self.predictor.eval()
        self.target.eval()

        N = observations.shape[0]
        novelties = []

        # 转为 Tensor (如果还不是)
        if isinstance(observations, np.ndarray):
            obs_tensor = torch.FloatTensor(observations)
        else:
            obs_tensor = observations

        with torch.no_grad():
            for i in range(0, N, batch_size):
                # 取出一个 batch
                batch_obs = obs_tensor[i: min(i + batch_size, N)].to(self.device)

                # 前向传播
                target_feat = self.target(batch_obs)
                pred_feat = self.predictor(batch_obs)

                # 计算 MSE (不平均，保留每个样本的误差)
                # shape: (batch_size, )
                error = F.mse_loss(pred_feat, target_feat, reduction='none').mean(dim=1)

                novelties.append(error.cpu())

        # 拼接回一个完整的 Tensor
        return torch.cat(novelties, dim=0).unsqueeze(1)  # shape (N, 1)


    def train_conflict_aware(self, online_states, offline_states):
        """
        [创新点] 冲突感知自适应更新 (Conflict-Aware Adaptive Update)

        结合了:
        1. 梯度投影 (Gradient Projection): 防止在线更新破坏离线记忆
        2. 自适应方差加权 (Variance-based Scaling): 根据数据稳定性调整步长

        参考文献思路:
        - Efficient and Stable Offline-to-online RL (IJCAI 2024)
        - Adaptive Regularization for Safe Control (OpenReview)
        """
        self.optimizer.zero_grad()

        # ==========================================
        # 1. 分别计算两个任务的 Loss
        # ==========================================

        # 任务 A: 适应在线新数据
        pred_on = self.predictor(online_states)
        target_on = self.target(online_states)
        loss_on = F.mse_loss(pred_on, target_on)

        # 任务 B: 保持离线旧记忆 (Anchor)
        pred_off = self.predictor(offline_states)
        target_off = self.target(offline_states)
        loss_off = F.mse_loss(pred_off, target_off)

        # ==========================================
        # 2. 计算梯度 (但不更新)
        # ==========================================

        # 获取 predictor 的参数
        params = list(self.predictor.parameters())

        # 计算在线梯度 g_on
        grad_on = torch.autograd.grad(loss_on, params, retain_graph=True)
        # 计算离线梯度 g_off
        grad_off = torch.autograd.grad(loss_off, params, retain_graph=False)

        # 展平梯度以便计算点积
        g_on_flat = torch.cat([g.view(-1) for g in grad_on])
        g_off_flat = torch.cat([g.view(-1) for g in grad_off])

        # ==========================================
        # 3. 冲突检测与梯度投影 (Project Conflicting Gradients)
        # ==========================================

        # 计算余弦相似度或点积
        dot_product = torch.dot(g_on_flat, g_off_flat)

        if dot_product < 0:
            # 发生冲突！在线更新的方向会增加离线数据的 Error
            # 解决方案：将 g_on 投影到 g_off 的法平面上
            # 公式: g_on_proj = g_on - ( (g_on . g_off) / (g_off . g_off) ) * g_off

            denom = torch.dot(g_off_flat, g_off_flat) + 1e-8
            projection_scalar = dot_product / denom

            # 对每个参数的梯度执行投影修正
            final_grads = []
            for g1, g2 in zip(grad_on, grad_off):
                g_proj = g1 - projection_scalar * g2
                # 即使投影后，我们还是希望混合一点离线梯度来进一步降低离线 Error
                final_grads.append(g_proj + g2)
        else:
            # 没有冲突，两个任务方向一致，直接相加
            final_grads = [g1 + g2 for g1, g2 in zip(grad_on, grad_off)]

        # ==========================================
        # 4. 幅度自适应 (基于不确定性/方差)
        # ==========================================

        # 如果在线数据的 Error 方差很大，说明样本极不稳定，应该减小更新步长
        # 计算在线 Error 的方差
        with torch.no_grad():
            raw_errors = F.mse_loss(pred_on, target_on, reduction='none').mean(dim=1)
            error_var = torch.var(raw_errors)
            # 自适应系数: 方差越大，alpha 越小
            # 这种动态调整符合 "Adaptive Learning Rate Scheduling" 的思想
            adaptive_scale = 1.0 / (1.0 + error_var.item())

            # ==========================================
        # 5. 应用梯度
        # ==========================================
        for param, grad in zip(self.predictor.parameters(), final_grads):
            if grad is not None:
                param.grad = grad * adaptive_scale  # 应用自适应缩放

        # 梯度裁剪 (常规操作)
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=1.0)
        self.optimizer.step()

        # return loss_on.item(), loss_off.item(), adaptive_scale