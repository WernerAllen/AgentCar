"""
PPO微调脚本 - 课程学习后在完整场景微调
使用Stable-Baselines3进行训练

用法:
  python finetune_ppo.py --finetune --pretrain outputs/curriculum_xxx/stage5_xxx/best_model.zip
  python finetune_ppo.py --eval models/xxx/best_model.zip
"""

import os
import torch
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

from formation_rl_env import FormationRLEnv


class CombinedExtractor(BaseFeaturesExtractor):
    """
    自定义特征提取器 - 处理Dict观测空间
    输入: {"image": (4, 84, 84), "vector": (72,)}  # 3帧×24维
    """
    
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # 图像分支: CNN
        image_space = observation_space.spaces["image"]
        n_input_channels = image_space.shape[0]  # 4
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 计算CNN输出维度
        with torch.no_grad():
            sample = torch.zeros(1, *image_space.shape)
            cnn_output_dim = self.cnn(sample).shape[1]
        
        self.image_linear = nn.Sequential(
            nn.Linear(cnn_output_dim, 128),
            nn.ReLU(),
        )
        
        # 向量分支: MLP
        vector_space = observation_space.spaces["vector"]
        vector_dim = vector_space.shape[0]  # 20
        
        self.vector_mlp = nn.Sequential(
            nn.Linear(vector_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations) -> torch.Tensor:
        # 提取图像特征
        image_features = self.image_linear(self.cnn(observations["image"]))
        
        # 提取向量特征
        vector_features = self.vector_mlp(observations["vector"])
        
        # 融合
        combined = torch.cat([image_features, vector_features], dim=1)
        return self.fusion(combined)


class FinetuneCallback(BaseCallback):
    """微调进度回调 - 显示详细训练进度"""
    
    def __init__(self, total_timesteps: int, print_interval: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_interval = print_interval
        self.last_print_step = 0
        
    def _on_step(self) -> bool:
        # 每隔一定步数打印进度
        if self.num_timesteps - self.last_print_step >= self.print_interval:
            self.last_print_step = self.num_timesteps
            progress = self.num_timesteps / self.total_timesteps * 100
            
            # 获取最近的episode统计
            mean_reward = 0.0
            mean_length = 0
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                mean_length = int(np.mean([ep['l'] for ep in self.model.ep_info_buffer]))
            
            print(f"[Finetune] Step {self.num_timesteps:,}/{self.total_timesteps:,} "
                  f"({progress:.1f}%) | Reward: {mean_reward:.1f} | EpLen: {mean_length}")
        return True


def make_env(scenario: str = "main", rank: int = 0, seed: int = 0):
    """创建环境工厂函数"""
    def _init():
        env = FormationRLEnv(scenario=scenario, num_cars=4, max_steps=2000)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def finetune(
    pretrain_path: str,
    total_timesteps: int = 500_000,
    n_envs: int = 8,            # 微调时适中的并行数
    scenario: str = "main",     # 完整场景
    save_dir: str = "models",
    log_dir: str = "logs",
    seed: int = 42,
    # PPO超参数 - 微调专用（更保守）
    learning_rate: float = 3e-5,  # 微调用更低学习率
    batch_size: int = 256,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.15,     # 微调时更小的裁剪范围
    ent_coef: float = 0.01,       # 微调时减少探索
    device: str = "auto",
):
    """
    课程学习后微调主函数
    
    Args:
        pretrain_path: 预训练模型路径（课程学习后的best_model）
        total_timesteps: 微调总步数
        n_envs: 并行环境数量
        scenario: 微调场景（默认main完整场景）
        save_dir: 模型保存目录
        log_dir: 日志目录
        seed: 随机种子
        learning_rate: 学习率（微调用更低值）
        batch_size: 批次大小
        n_epochs: 每次更新的epoch数
        gamma: 折扣因子
        gae_lambda: GAE参数
        clip_range: PPO裁剪范围（微调用更小值）
        ent_coef: 熵系数（微调用更小值减少探索）
        device: 训练设备
    """
    
    # 检查预训练模型是否存在
    model_file = pretrain_path if pretrain_path.endswith('.zip') else pretrain_path + '.zip'
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"预训练模型不存在: {model_file}")
    
    # 创建目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"ppo_formation_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 60)
    print("PPO Formation Control Training")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"Scenario: {scenario}")
    print(f"Device: {device}")
    print(f"Save path: {save_path}")
    print("=" * 60)
    
    # 创建并行环境
    # 使用SubprocVecEnv充分利用多核CPU
    if n_envs > 1:
        env = SubprocVecEnv([make_env(scenario, i, seed) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(scenario, 0, seed)])
    
    # 创建评估环境
    eval_env = DummyVecEnv([make_env(scenario, 0, seed + 100)])
    
    # 加载预训练模型进行微调
    print(f"\n加载预训练模型: {pretrain_path}")
    model = PPO.load(pretrain_path, env=env, device=device)
    
    # 更新微调专用超参数
    model.learning_rate = learning_rate
    model.clip_range = lambda _: clip_range
    model.ent_coef = ent_coef
    model.n_epochs = n_epochs
    model.batch_size = batch_size
    
    print(f"微调超参数:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Clip range: {clip_range}")
    print(f"  Entropy coef: {ent_coef}")
    print(f"\nModel architecture:")
    print(model.policy)
    
    # 回调函数
    callbacks = [
        # 定期保存检查点
        CheckpointCallback(
            save_freq=10000 // n_envs,
            save_path=save_path,
            name_prefix="ppo_formation"
        ),
        # 评估回调
        EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=20000 // n_envs,  # 降低评估频率提升训练速度
            n_eval_episodes=5,
            deterministic=True,
        ),
        # 微调进度监控
        FinetuneCallback(total_timesteps=total_timesteps, print_interval=5000),
    ]
    
    # 开始训练
    print("\n开始微调训练...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    # 保存最终模型
    final_path = os.path.join(save_path, "final_model")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")
    
    # 关闭环境
    env.close()
    eval_env.close()
    
    return model, save_path


def evaluate(model_path: str, n_episodes: int = 10, render: bool = False):
    """评估训练好的模型"""
    
    print(f"\nEvaluating model: {model_path}")
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 创建评估环境
    env = FormationRLEnv(scenario="main", num_cars=4, max_steps=2000)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # 检查是否成功到达终点
        if all(car.x >= env.env_params.goal_x for car in env.cars):
            success_count += 1
        
        print(f"Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}, "
              f"avg_x={info['avg_car_x']:.1f}")
    
    env.close()
    
    print("\n" + "=" * 40)
    print(f"Evaluation Results ({n_episodes} episodes):")
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean Length: {np.mean(episode_lengths):.0f} ± {np.std(episode_lengths):.0f}")
    print(f"  Success Rate: {success_count / n_episodes * 100:.1f}%")
    print("=" * 40)
    
    return episode_rewards, episode_lengths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PPO微调脚本 - 课程学习后在完整场景微调",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 微调课程学习后的模型
  python train_ppo.py --finetune --pretrain outputs/curriculum_xxx/stage5_xxx/best_model.zip
  
  # 评估模型
  python train_ppo.py --eval models/xxx/best_model.zip
  
  # 指定场景和步数
  python train_ppo.py --finetune --pretrain model.zip --scenario main --timesteps 500000
        """
    )
    parser.add_argument("--finetune", action="store_true", help="运行微调训练")
    parser.add_argument("--pretrain", type=str, default=None, help="预训练模型路径（课程学习后的best_model）")
    parser.add_argument("--eval", type=str, default=None, help="评估模型路径")
    parser.add_argument("--scenario", type=str, default="main", help="微调场景（默认main完整场景）")
    parser.add_argument("--timesteps", type=int, default=500_000, help="微调总步数")
    parser.add_argument("--n_envs", type=int, default=8, help="并行环境数量")
    parser.add_argument("--device", type=str, default="auto", help="训练设备 (cpu/cuda/auto)")
    
    args = parser.parse_args()
    
    if args.eval:
        evaluate(args.eval)
    elif args.finetune:
        if not args.pretrain:
            print("错误: --finetune 需要指定 --pretrain 预训练模型路径")
            print("用法: python train_ppo.py --finetune --pretrain <模型路径>")
            exit(1)
        finetune(
            pretrain_path=args.pretrain,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            scenario=args.scenario,
            device=args.device,
        )
    else:
        # 显示帮助信息
        parser.print_help()
        print("\n提示: 使用 --finetune --pretrain <模型路径> 开始微调")
