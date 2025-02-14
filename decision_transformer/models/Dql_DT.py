import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_atoms=51, v_min=-10, v_max=10):
        super(Critic, self).__init__()

        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, num_atoms)

        self.q1_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, num_atoms)  # 输出 Q 值的概率分布
        )

        self.q2_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, num_atoms)  # 输出 Q 值的概率分布
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        q1_dist = F.softmax(self.q1_model(x), dim=-1)  # 归一化为概率分布
        q2_dist = F.softmax(self.q2_model(x), dim=-1)
        return q1_dist, q2_dist

    def q_min(self, state, action):
        q1_dist, q2_dist = self.forward(state, action)
        q1 = (q1_dist * self.support.to(state.device)).sum(dim=-1)  # 计算期望 Q 值
        q2 = (q2_dist * self.support.to(state.device)).sum(dim=-1)
        return torch.min(q1, q2)
    
    def min(self, q1_dist, q2_dist):
        q1_mean = (q1_dist * self.support.to(q1_dist.device)).sum(dim=-1)  # [B]
        q2_mean = (q2_dist * self.support.to(q2_dist.device)).sum(dim=-1)  # [B]
        target_q_dist = torch.where(q1_mean.unsqueeze(-1) < q2_mean.unsqueeze(-1), q1_dist, q2_dist)
        return target_q_dist
    
    def compute_target_distribution(self, rewards, dones, target_q_dist, discount):
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)

        batch_size, T = rewards.shape  # 确保 rewards 形状正确
        num_atoms = self.num_atoms

        # 如果 target_q_dist 形状是 (B, 51)，扩展为 (B, T, 51)
        if target_q_dist.dim() == 2 and target_q_dist.shape[1] == num_atoms:
            target_q_dist = target_q_dist.unsqueeze(1).expand(-1, T, -1)  # [B, 1, 51] -> [B, T, 51]


        # 计算目标支持集的 Q 值
        z_target = rewards.unsqueeze(-1) + discount * self.support.to(rewards.device).view(1, 1, -1) * (1 - dones.unsqueeze(-1))
        z_target = z_target.clamp(self.v_min, self.v_max)

        # 计算投影索引
        b = (z_target - self.v_min) / (self.v_max - self.v_min) * (num_atoms - 1)
        lower_idx = b.floor().long()
        upper_idx = b.ceil().long()

        lower_idx = lower_idx.clamp(0, num_atoms - 1)
        upper_idx = upper_idx.clamp(0, num_atoms - 1)

        # 初始化目标分布
        target_distribution = torch.zeros(batch_size, T, num_atoms, device=rewards.device)

        # 计算线性插值加权分布
        offset = torch.linspace(0, (batch_size - 1) * T * num_atoms, batch_size * T, dtype=torch.long, device=rewards.device).view(batch_size, T, 1)

        # **调试信息**
        # print(f"target_q_dist.shape: {target_q_dist.shape}")  # 可能是 (B, T, 51)
        # print(f"target_distribution.shape: {target_distribution.shape}")  # 可能是 (B, T, 51)
        # print(f"lower_idx.shape: {lower_idx.shape}, upper_idx.shape: {upper_idx.shape}")  # 可能是 (B, T, 51)
        # print(f"offset.shape: {offset.shape}")  # 可能是 (B, T, 1)

        # **确保 target_q_dist 维度匹配**
        if target_q_dist.shape != (batch_size, T, num_atoms):
            raise ValueError(f"target_q_dist shape mismatch! Expected {(batch_size, T, num_atoms)}, got {target_q_dist.shape}")

        target_distribution.view(-1).index_add_(0, (lower_idx + offset).view(-1), (target_q_dist * (upper_idx.float() - b)).view(-1))
        target_distribution.view(-1).index_add_(0, (upper_idx + offset).view(-1), (target_q_dist * (b - lower_idx.float())).view(-1))

        return target_distribution



class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            sar=False,
            scale=1.,
            rtg_no_q=False,
            infer_no_q=False,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.config = config
        self.sar = sar
        self.scale = scale
        self.rtg_no_q = rtg_no_q
        self.infer_no_q = infer_no_q

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_rewards = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_rewards = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards=None, targets=None, returns_to_go=None, timesteps=None, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        reward_embeddings = self.embed_rewards(rewards / self.scale)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        reward_embeddings = reward_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        if self.sar:
            stacked_inputs = torch.stack(
                (state_embeddings, action_embeddings, reward_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        else:
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        if self.sar:
            action_preds = self.predict_action(x[:, 0])
            rewards_preds = self.predict_rewards(x[:, 1])
            state_preds = self.predict_state(x[:, 2])
        else:
            action_preds = self.predict_action(x[:, 1])
            state_preds = self.predict_state(x[:, 2])
            rewards_preds = None


        return state_preds, action_preds, rewards_preds

    def get_action(self, critic, states, actions, rewards=None, returns_to_go=None, timesteps=None, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim).repeat_interleave(repeats=50, dim=0)
        actions = actions.reshape(1, -1, self.act_dim).repeat_interleave(repeats=50, dim=0)
        rewards = rewards.reshape(1, -1, 1).repeat_interleave(repeats=50, dim=0)
        timesteps = timesteps.reshape(1, -1).repeat_interleave(repeats=50, dim=0)

        bs = returns_to_go.shape[0]
        returns_to_go = returns_to_go.reshape(bs, -1, 1).repeat_interleave(repeats=50 // bs, dim=0)
        returns_to_go = torch.cat([returns_to_go, torch.randn((50-returns_to_go.shape[0], returns_to_go.shape[1], 1), device=returns_to_go.device)], dim=0)
            

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            rewards = rewards[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # padding
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1).repeat_interleave(repeats=50, dim=0)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
            rewards = torch.cat(
                [torch.zeros((rewards.shape[0], self.max_length-rewards.shape[1], 1), device=rewards.device), rewards],
                dim=1
            ).to(dtype=torch.float32)

            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length-actions.shape[1], self.act_dim), device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
        else:
            attention_mask = None

        returns_to_go[bs:, -1] = returns_to_go[bs:, -1] + torch.randn_like(returns_to_go[bs:, -1]) * 0.1
        if not self.rtg_no_q:
            returns_to_go[-1, -1] = critic.q_min(states[-1:, -2], actions[-1:, -2]).flatten() - rewards[-1, -2] / self.scale
        _, action_preds, return_preds = self.forward(states, actions, rewards, None, returns_to_go=returns_to_go, timesteps=timesteps, attention_mask=attention_mask, **kwargs)
    
        
        state_rpt = states[:, -1, :]
        action_preds = action_preds[:, -1, :]

        q_value = critic.q_min(state_rpt, action_preds).flatten()
        idx = torch.multinomial(F.softmax(q_value, dim=-1), 1)

        if not self.infer_no_q:
            return action_preds[idx]
        else:
            return action_preds[0]
