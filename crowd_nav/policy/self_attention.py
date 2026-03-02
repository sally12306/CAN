import torch
import torch.nn as nn
from crowd_nav.policy.cadrl import mlp


class SelfAttention(nn.Module):
    def __init__(
            self, input_dim=13, self_state_dim=6, mlp1_dims=[150, 100], mlp2_dims=[100, 50],
            mlp3_dims=[150, 100, 100, 1],
            attention_dims=[100, 100, 1], with_global_state=True,
            cell_size=None, cell_num=None
    ):
        super().__init__()
        self.action_dim = 2
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim + self.action_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = state.reshape(-1, 65)
        state = state.reshape(state.shape[0], 5, 13)
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)). \
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        return joint_state


if __name__ == '__main__':
    # self_attention = SelfAttention()
    # x = torch.zeros(256, 5, 78)
    # print(x.shape)
    # y = self_attention(x)
    # print(y.shape)
    base_network = nn.Sequential(
        nn.Linear(56, 256),
        nn.ReLU(),
        nn.Linear(256, 256), )
    tensor = torch.randn(10, 20, 30, 56)
    print(base_network(tensor).shape)
