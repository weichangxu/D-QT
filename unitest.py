import torch

def compute_target_distribution(rewards, dones, target_q_dist, discount, num_atoms, support, v_min, v_max):
    rewards = rewards.squeeze(-1)
    dones = dones.squeeze(-1)

    batch_size, T = rewards.shape  # 确保 rewards 形状正确

    # 如果 target_q_dist 形状是 (B, 51)，扩展为 (B, T, 51)
    if target_q_dist.dim() == 2 and target_q_dist.shape[1] == num_atoms:
        target_q_dist = target_q_dist.unsqueeze(1).expand(-1, T, -1)  # [B, 1, 51] -> [B, T, 51]

    # 计算目标支持集的 Q 值
    z_target = rewards.unsqueeze(-1) + discount * support.to(rewards.device).view(1, 1, -1) * (1 - dones.unsqueeze(-1))
    z_target = z_target.clamp(v_min, v_max)

    # 计算投影索引
    b = (z_target - v_min) / (v_max - v_min) * (num_atoms - 1)
    lower_idx = b.floor().long()
    upper_idx = b.ceil().long()

    lower_idx = lower_idx.clamp(0, num_atoms - 1)
    upper_idx = upper_idx.clamp(0, num_atoms - 1)

    # 初始化目标分布
    target_distribution = torch.zeros(batch_size, T, num_atoms, device=rewards.device)

    # 计算线性插值加权分布
    offset = torch.linspace(0, (batch_size - 1) * T * num_atoms, batch_size * T, dtype=torch.long, device=rewards.device).view(batch_size, T, 1)
    target_distribution.view(-1).index_add_(0, (lower_idx + offset).view(-1), (target_q_dist * (upper_idx.float() - b)).view(-1))
    target_distribution.view(-1).index_add_(0, (upper_idx + offset).view(-1), (target_q_dist * (b - lower_idx.float())).view(-1))

    return target_distribution

import unittest
import torch

class TestComputeTargetDistribution(unittest.TestCase):

    def setUp(self):
        # 准备测试数据
        self.rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape: [B, T]
        self.dones = torch.tensor([[0.0, 0.0], [0.0, 0.0]])    # shape: [B, T]
        self.target_q_dist = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                                           [0.5, 0.4, 0.3, 0.2, 0.1]])  # shape: [B, num_atoms]
        self.discount = 0.99
        self.num_atoms = 5
        self.support = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0])  # size: num_atoms
        self.v_min = -1.0
        self.v_max = 3.0

    def test_compute_target_distribution(self):
        # 调用被测试函数
        target_distribution = compute_target_distribution(self.rewards, self.dones, self.target_q_dist, 
                                                          self.discount, self.num_atoms, self.support, 
                                                          self.v_min, self.v_max)
        print(target_distribution)
        # 断言输出形状正确
        self.assertEqual(target_distribution.shape, (2, 2, 5))  # Batch size 2, T=2, num_atoms=5
        
        # 可以添加其他断言来验证具体数值是否符合预期
        # 示例：检查某个值是否在合理范围内
        self.assertTrue((target_distribution >= 0).all())
        self.assertTrue((target_distribution <= 1).all())

if __name__ == '__main__':
    unittest.main()
