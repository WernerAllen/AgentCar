"""
行为克隆预训练：让策略学会在empty场景输出零action

重要：必须使用与train_ppo.py相同的CombinedExtractor网络结构！
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from formation_rl_env import FormationRLEnv
from train_ppo import CombinedExtractor  # 导入自定义特征提取器


def collect_expert_data(n_episodes=100, max_steps=200):
    """收集专家数据：零action保持编队
    
    专家策略：
    - 输出零action（保持当前编队）
    - 让DWA自然追踪理想位置
    - 横向修正能力留给后续RL训练学习
    
    理由：P控制器可能导致车辆过度修正相撞
    """
    env = FormationRLEnv(scenario='empty', num_cars=4, max_steps=max_steps, use_rl=True)
    
    observations = {'image': [], 'vector': []}
    actions = []
    
    for ep in range(n_episodes):
        # 显示进度
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Episode {ep + 1}/{n_episodes}...")
        
        obs, _ = env.reset(seed=ep)
        for step in range(max_steps):
            # 专家action：零（保持编队，不主动修正）
            expert_action = np.zeros(4, dtype=np.float32)
            
            observations['image'].append(obs['image'])
            observations['vector'].append(obs['vector'])
            actions.append(expert_action.copy())
            
            obs, _, term, trunc, _ = env.step(expert_action)
            if term or trunc:
                break
    
    env.close()
    return {
        'image': np.array(observations['image']),
        'vector': np.array(observations['vector']),
        'actions': np.array(actions)
    }


def pretrain_policy(model, expert_data, epochs=50, batch_size=64, lr=1e-4):
    """用行为克隆预训练策略"""
    device = model.device
    
    # 准备数据
    images = torch.FloatTensor(expert_data['image']).to(device)
    vectors = torch.FloatTensor(expert_data['vector']).to(device)
    actions = torch.FloatTensor(expert_data['actions']).to(device)
    
    dataset = TensorDataset(images, vectors, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 优化器
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"预训练开始: {len(dataset)} 样本, {epochs} epochs")
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_img, batch_vec, batch_act in dataloader:
            # 构造观测字典
            obs = {'image': batch_img, 'vector': batch_vec}
            
            # 前向传播
            features = model.policy.extract_features(obs)
            if model.policy.share_features_extractor:
                latent_pi, _ = model.policy.mlp_extractor(features)
            else:
                pi_features = model.policy.pi_features_extractor(obs)
                latent_pi, _ = model.policy.mlp_extractor(pi_features)
            
            # 获取动作均值
            mean_actions = model.policy.action_net(latent_pi)
            
            # 计算损失
            loss = criterion(mean_actions, batch_act)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return model


def main():
    # 创建环境和模型
    env = FormationRLEnv(scenario='empty', num_cars=4, max_steps=2000, use_rl=True)
    
    # 策略网络配置 - 必须与train_ppo.py和curriculum_train.py一致！
    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128], vf=[256, 128])
    )
    
    # 创建PPO模型（使用与训练脚本相同的网络结构）
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,  # 关键：使用相同的网络结构
        verbose=0,
        device='cuda'
    )
    
    # 收集专家数据（学习"输出零"不需要太多数据）
    print("收集专家数据...")
    expert_data = collect_expert_data(n_episodes=100, max_steps=100)  # 100x100=10000样本足够
    print(f"收集了 {len(expert_data['actions'])} 个样本")
    
    # 预训练（100 epochs足够学会输出零）
    model = pretrain_policy(model, expert_data, epochs=100, batch_size=128)
    
    # 保存预训练模型
    model.save("models/pretrain_bc")
    print("预训练模型已保存到 models/pretrain_bc")
    
    # 测试
    print("\n测试预训练模型:")
    obs, _ = env.reset(seed=0)
    for step in range(10):
        action, _ = model.predict(obs, deterministic=True)
        print(f"Step {step}: action={np.round(action, 3)}")
        obs, _, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
