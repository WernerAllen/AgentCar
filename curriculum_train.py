"""
课程学习训练脚本
自动按阶段执行训练，每阶段结束后可视化测试

使用方法:
1. 先运行 pretrain_bc.py 进行empty场景的监督学习预训练
2. 再运行本脚本进行有障碍物场景的RL训练

目录结构:
outputs/
├── curriculum_YYYYMMDD_HHMMSS/    # 单次课程学习的所有输出
│   ├── stage1_s4_center_small/    # 阶段1(入门避障)
│   │   ├── best_model.zip
│   │   ├── final_model.zip
│   │   ├── evaluations.npz
│   │   └── test_result.png
│   ├── stage2_s1s2_mixed/         # 阶段2(双向避障-左右交替)
│   │   └── ...
│   ├── stage3_s3_narrow/          # 阶段3(窄道)
│   │   └── ...
│   ├── logs/                      # TensorBoard日志
│   │   └── PPO_*/
│   └── summary.txt                # 训练总结
"""
import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from formation_rl_env import FormationRLEnv
from train_ppo import CombinedExtractor, make_env


# 课程定义：(场景名, 训练步数, 描述)
# 注意：empty场景通过pretrain_bc.py单独进行监督学习预训练
# 难度递进：入门→泛化→分流→窄道→极窄
# 调整：S5(分流)移到S3(窄道)前，因为分流不需要收缩队形，难度更低
# 
# 训练步数计算：n_envs=8, n_steps=2048 → 每16384步一次PPO更新
# 建议每阶段至少15-30次更新，即250k-500k步
CURRICULUM = [
    ("s4_center_small", 500000, "入门：小障碍物，队形不变通过"),      # ~30次更新
    ("s1s2_mixed", 800000, "泛化：左右交替障碍，学会双向避障"),       # ~48次更新
    ("s5_center_large", 500000, "分流：中间大障碍，分两边通过"),      # ~30次更新（不需收缩）
    ("s3_narrow", 1000000, "窄道：两侧障碍，收缩队形通过"),          # ~61次更新
    ("s6_very_narrow", 1500000, "极窄：一字长蛇阵纵队通过"),         # ~91次更新
]


def setup_output_dir(base_dir: str = "outputs") -> str:
    """创建有组织的输出目录结构"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"curriculum_{timestamp}")
    
    # 创建子目录
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    
    for i, (scenario, _, _) in enumerate(CURRICULUM):
        stage_dir = os.path.join(run_dir, f"stage{i+1}_{scenario}")
        os.makedirs(stage_dir, exist_ok=True)
    
    return run_dir

class CurriculumCallback(BaseCallback):
    """课程学习回调：记录训练进度并实时显示详细诊断信息"""
    def __init__(self, stage_name: str, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.stage_name = stage_name
        self.total_timesteps = total_timesteps
        self.rewards = []
        self.last_print_step = 0
        self.print_interval = 20000  # 每2万步打印一次（与诊断同步）
        self.last_detailed_print = 0
        self.detailed_interval = 20000  # 每2万步打印详细诊断
        
        # 诊断统计
        self.collision_count = 0
        self.success_count = 0
        self.episode_count = 0
        self.last_infos = []  # 存储最近的info用于诊断
        
        # #15修复：早停机制
        self.early_stop_patience = 100000  # 连续10万步无改善则警告
        self.best_mean_reward = -float('inf')
        self.steps_without_improvement = 0
        
    def _on_step(self) -> bool:
        # 收集info信息用于诊断
        if hasattr(self, 'locals') and 'infos' in self.locals:
            for info in self.locals['infos']:
                if info.get('collision', False):
                    self.collision_count += 1
                if info.get('goal_reached', False):
                    self.success_count += 1
                # 存储最近的info
                if len(self.last_infos) < 100:
                    self.last_infos.append(info)
                else:
                    self.last_infos.pop(0)
                    self.last_infos.append(info)
        
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
            
            print(f"[{self.stage_name}] Step {self.num_timesteps:,}/{self.total_timesteps:,} "
                  f"({progress:.1f}%) | Reward: {mean_reward:.1f} | EpLen: {mean_length}")
            
            # #15修复：早停检测
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.steps_without_improvement = 0
            else:
                self.steps_without_improvement += self.print_interval
            
            if self.steps_without_improvement >= self.early_stop_patience:
                print(f"  ⚠️ WARNING: No improvement for {self.steps_without_improvement:,} steps!")
                print(f"  ⚠️ Consider checking training parameters or stopping early.")
        
        # 每隔较长步数打印详细诊断
        if self.num_timesteps - self.last_detailed_print >= self.detailed_interval:
            self.last_detailed_print = self.num_timesteps
            self._print_detailed_diagnostics()
        
        return True
    
    def _print_detailed_diagnostics(self):
        """打印详细诊断信息"""
        print(f"\n{'='*60}")
        print(f"[{self.stage_name}] 诊断报告 @ {self.num_timesteps:,} steps")
        print(f"{'='*60}")
        
        # 碰撞/成功统计
        total_eps = self.collision_count + self.success_count
        if total_eps > 0:
            collision_rate = self.collision_count / max(1, total_eps) * 100
            success_rate = self.success_count / max(1, total_eps) * 100
            print(f"碰撞率: {collision_rate:.1f}% ({self.collision_count}/{total_eps})")
            print(f"成功率: {success_rate:.1f}% ({self.success_count}/{total_eps})")
        
        # 碰撞位置分析（从info中提取）
        collision_xs = [info.get('collision_x', -1) for info in self.last_infos 
                        if info.get('collision', False) and 'collision_x' in info]
        if collision_xs:
            avg_collision_x = np.mean(collision_xs)
            print(f"平均碰撞位置x: {avg_collision_x:.1f}m (第一个窄门在x=25m)")
        
        # 碰撞时车辆y坐标分析
        collision_ys_list = [info.get('collision_ys', []) for info in self.last_infos 
                            if info.get('collision', False) and 'collision_ys' in info]
        if collision_ys_list:
            all_ys = [y for ys in collision_ys_list for y in ys]
            if all_ys:
                print(f"碰撞时车辆y范围: [{min(all_ys):.2f}, {max(all_ys):.2f}]m (理想收缩后应在±0.43m)")
        
        # 通道与队形收缩统计
        passage_widths = [info.get('passage_width') for info in self.last_infos if 'passage_width' in info]
        passage_centers = [info.get('passage_center') for info in self.last_infos if 'passage_center' in info]
        scale_factors = [info.get('scale_factor') for info in self.last_infos if 'scale_factor' in info]
        if passage_widths:
            print(f"通道宽度: mean={np.mean(passage_widths):.2f}m | min={np.min(passage_widths):.2f}m")
        if passage_centers:
            print(f"通道中心y: mean={np.mean(passage_centers):.2f}m")
        if scale_factors:
            print(f"缩放因子: mean={np.mean(scale_factors):.2f} | min={np.min(scale_factors):.2f}")

        # 目标点安全修正统计（是否经常被“改写”）
        adjust_means = [info.get('target_adjustment_mean') for info in self.last_infos if 'target_adjustment_mean' in info]
        adjust_maxs = [info.get('target_adjustment_max') for info in self.last_infos if 'target_adjustment_max' in info]
        adjust_counts = [info.get('target_adjustment_count') for info in self.last_infos if 'target_adjustment_count' in info]
        if adjust_counts:
            adjust_rate = np.mean([cnt > 0 for cnt in adjust_counts]) * 100
            print(f"目标修正: 触发率={adjust_rate:.1f}% | 平均修正幅度={np.mean(adjust_means):.3f} | 最大修正幅度={np.max(adjust_maxs):.3f}")

        # DWA fallback统计（无可行解时会增大）
        fallback_counts = [info.get('dwa_fallback_count') for info in self.last_infos if 'dwa_fallback_count' in info]
        if fallback_counts:
            fallback_rate = np.mean([cnt > 0 for cnt in fallback_counts]) * 100
            print(f"DWA fallback: 触发率={fallback_rate:.1f}% | 平均次数/步={np.mean(fallback_counts):.2f}")

        # 最小距离监控（用于判断是否长期贴边/贴障）
        min_obstacle_dists = [info.get('min_obstacle_dist') for info in self.last_infos if 'min_obstacle_dist' in info]
        min_boundary_dists = [info.get('min_boundary_dist') for info in self.last_infos if 'min_boundary_dist' in info]
        min_car_dists = [info.get('min_car_dist') for info in self.last_infos if 'min_car_dist' in info]
        if min_obstacle_dists:
            print(f"最小障碍距离: mean={np.mean(min_obstacle_dists):.2f} | min={np.min(min_obstacle_dists):.2f}")
        if min_boundary_dists:
            print(f"最小边界距离: mean={np.mean(min_boundary_dists):.2f} | min={np.min(min_boundary_dists):.2f}")
        if min_car_dists:
            print(f"最小车车距离: mean={np.mean(min_car_dists):.2f} | min={np.min(min_car_dists):.2f}")

        # 碰撞类型统计
        collision_types = [info.get('collision_type', 'unknown') for info in self.last_infos 
                           if info.get('collision', False)]
        if collision_types:
            from collections import Counter
            type_counts = Counter(collision_types)
            type_str = ', '.join([f"{t}:{c}" for t, c in type_counts.items()])
            print(f"碰撞类型分布: {type_str}")
            
            # 车车碰撞时的最小距离
            car_dists = [info.get('min_car_dist', 999) for info in self.last_infos 
                         if info.get('collision_type') == 'car_car']
            if car_dists:
                print(f"车车碰撞时最小距离: {np.mean(car_dists):.3f}m (安全距离>0.54m)")
        
        # 从最近的info中提取关键指标
        if self.last_infos:
            # 平均各项惩罚
            keys = ['target_out_of_bounds_penalty', 'formation_penalty', 
                    'obstacle_proximity_penalty', 'car_car_proximity_penalty',
                    'shape_error', 'scale_error', 'position_error']
            
            for key in keys:
                values = [info.get(key, 0) for info in self.last_infos if key in info]
                if values:
                    avg_val = np.mean(values)
                    print(f"  {key}: {avg_val:.3f}")
            
            # 危险因子
            alphas = [info.get('danger_alpha', 0) for info in self.last_infos if 'danger_alpha' in info]
            if alphas:
                print(f"  danger_alpha(平均): {np.mean(alphas):.3f}")
        
        # 重置统计（用于下一个周期）
        self.collision_count = 0
        self.success_count = 0
        self.last_infos = []
        print(f"{'='*60}\n")
    
    def _on_rollout_end(self):
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
            self.rewards.append(mean_reward)


def visualize_test(model_path: str, scenario: str, save_path: str, max_steps: int = 2000):
    """可视化测试模型"""
    model = PPO.load(model_path)
    env = FormationRLEnv(scenario=scenario, num_cars=4, max_steps=max_steps, use_rl=True)
    
    obs, _ = env.reset()
    trajectories = [[] for _ in range(4)]
    rewards = []
    
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        for i, car in enumerate(env.cars):
            trajectories[i].append((car.x, car.y))
        
        if terminated or truncated:
            break
    
    # 绘图：轨迹图，适度拉伸Y轴
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # 小车实际尺寸
    car_length = env.vehicle_params.car_length
    car_width = env.vehicle_params.car_width
    
    # 轨迹图：使用不同线型避免重叠遮挡
    colors = ['red', 'blue', 'green', 'orange']
    linestyles = ['-', '--', '-.', ':']
    for i, traj in enumerate(trajectories):
        xs, ys = zip(*traj)
        ax.plot(xs, ys, color=colors[i], linestyle=linestyles[i], 
                label=f'Car {i}', linewidth=2.5, alpha=0.85)
        
        # 绘制起点小车轮廓（矩形，实线边框）
        start_rect = plt.Rectangle(
            (xs[0] - car_length/2, ys[0] - car_width/2),
            car_length, car_width, 
            facecolor=colors[i], alpha=0.8, edgecolor='black', linewidth=1.5, zorder=5
        )
        ax.add_patch(start_rect)
        
        # 绘制终点小车轮廓（矩形，虚线边框）
        end_rect = plt.Rectangle(
            (xs[-1] - car_length/2, ys[-1] - car_width/2),
            car_length, car_width,
            facecolor=colors[i], alpha=0.5, edgecolor='black', linewidth=1.5, 
            linestyle='--', zorder=5
        )
        ax.add_patch(end_rect)
    
    # 绘制起点编队框（虚线矩形标注2x2编队）
    start_xs = [traj[0][0] for traj in trajectories]
    start_ys = [traj[0][1] for traj in trajectories]
    formation_rect = plt.Rectangle(
        (min(start_xs) - 0.5, min(start_ys) - 0.5),
        max(start_xs) - min(start_xs) + 1,
        max(start_ys) - min(start_ys) + 1,
        fill=False, edgecolor='purple', linewidth=2, linestyle=':', zorder=4,
        label='Formation'
    )
    ax.add_patch(formation_rect)
    
    # 绘制障碍物
    for obs in env.obstacles:
        if obs.width > 0:
            rect = plt.Rectangle(
                (obs.x - obs.width/2, obs.y - obs.height/2),
                obs.width, obs.height, color='gray', alpha=0.7
            )
            ax.add_patch(rect)
        else:
            circle = plt.Circle((obs.x, obs.y), obs.radius, color='gray', alpha=0.7)
            ax.add_patch(circle)
    
    # 边界
    road_hw = env.env_params.road_half_width
    ax.axhline(y=road_hw, color='black', linestyle='--', linewidth=2)
    ax.axhline(y=-road_hw, color='black', linestyle='--', linewidth=2)
    
    # 设置坐标轴范围
    max_x = max(max(traj, key=lambda p: p[0])[0] for traj in trajectories) + 10
    ax.set_xlim(-2, min(max_x, 120))  # 根据实际轨迹长度设置，最多120m
    ax.set_ylim(-road_hw - 0.5, road_hw + 0.5)  # Y轴适度扩展
    ax.set_aspect('equal')  # 保持真实比例，不变形
    
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.set_title(f'Trajectory - {scenario} (Total Reward: {sum(rewards):.1f})', fontsize=16)
    # 图例放在图外右侧
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 返回测试结果
    final_x = float(np.mean([car.x for car in env.cars]))
    collision = bool(info.get('collision', False))
    return {
        'final_x': final_x,
        'total_reward': float(sum(rewards)),
        'collision': collision,
        'steps': len(rewards)
    }


def train_stage(
    stage_idx: int,
    scenario: str,
    timesteps: int,
    description: str,
    run_dir: str,
    base_model_path: str = None,
    n_envs: int = 8
):
    """训练单个阶段"""
    print(f"\n{'='*70}")
    print(f"STAGE {stage_idx + 1}: {scenario} - {description}")
    print(f"Timesteps: {timesteps}, Base model: {base_model_path or 'None'}")
    print(f"{'='*70}\n")
    
    # 创建环境（暂时移除VecNormalize，奖励设计已合理）
    # 使用SubprocVecEnv充分利用多核CPU
    env = SubprocVecEnv([make_env(scenario, i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(scenario, 0)])
    
    # 使用预定义的阶段目录
    model_dir = os.path.join(run_dir, f"stage{stage_idx+1}_{scenario}")
    
    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128], vf=[256, 128])
    )
    
    # 检查模型文件（SB3会自动添加.zip后缀）
    model_file = base_model_path + ".zip" if base_model_path and not base_model_path.endswith(".zip") else base_model_path
    if base_model_path and os.path.exists(model_file):
        print(f"Loading base model from {base_model_path}")
        try:
            model = PPO.load(base_model_path, env=env)
            # v5.0修复：根据阶段调整学习率
            # Stage 1用较高学习率(5e-5)，后续阶段用更低学习率(3e-5)进行微调
            model.learning_rate = 3e-5
            print(f"  Model loaded, learning_rate set to {model.learning_rate}")
        except Exception as e:
            print(f"  WARNING: Failed to load model: {e}")
            print(f"  Creating new model instead...")
            base_model_path = None  # 回退到创建新模型
    # v5.0修复：处理模型加载失败的情况
    if base_model_path is None or not os.path.exists(model_file):
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            max_grad_norm=0.5,  # #13修复：显式设置梯度裁剪
            verbose=1,
            tensorboard_log=None  # 禁用tensorboard（如需启用请先安装：pip install tensorboard）
        )
    
    # 回调
    # #14修复：根据总步数动态调整评估频率，确保至少50次评估
    eval_freq = max(5000, min(10000, timesteps // 50))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=eval_freq,  # 动态评估频率
        n_eval_episodes=3,
        deterministic=True,
        verbose=1  # 打印每次评估结果
    )
    
    curriculum_callback = CurriculumCallback(scenario, timesteps)
    
    # 训练
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, curriculum_callback],
        progress_bar=False  # 禁用进度条（需要tqdm/rich）
    )
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, "final_model.zip")
    model.save(final_model_path)
    
    # 可视化测试
    best_model_path = os.path.join(model_dir, "best_model.zip")
    if not os.path.exists(best_model_path):
        best_model_path = final_model_path
    
    test_result = visualize_test(
        best_model_path, 
        scenario,
        os.path.join(model_dir, f"test_{scenario}.png")
    )
    
    print(f"\nStage {stage_idx + 1} Results:")
    print(f"  Final X: {test_result['final_x']:.1f}m")
    print(f"  Total Reward: {test_result['total_reward']:.1f}")
    print(f"  Collision: {test_result['collision']}")
    print(f"  Best model: {best_model_path}")
    
    env.close()
    eval_env.close()
    
    return best_model_path, test_result


def run_curriculum(
    start_stage: int = 0,
    base_model: str = None,
    n_envs: int = 8,
    output_dir: str = "outputs"
):
    """运行完整课程学习"""
    # 创建有组织的输出目录
    run_dir = setup_output_dir(output_dir)
    
    print("\n" + "="*70)
    print("CURRICULUM LEARNING")
    print(f"Output: {run_dir}")
    print("="*70)
    print("\nStages:")
    for i, (scenario, steps, desc) in enumerate(CURRICULUM):
        status = "SKIP" if i < start_stage else "TODO"
        print(f"  {i+1}. {scenario}: {steps} steps - {desc} [{status}]")
    print()
    
    current_model = base_model
    results = []
    
    for i, (scenario, timesteps, description) in enumerate(CURRICULUM):
        if i < start_stage:
            continue
        
        model_path, test_result = train_stage(
            stage_idx=i,
            scenario=scenario,
            timesteps=timesteps,
            description=description,
            run_dir=run_dir,
            base_model_path=current_model,
            n_envs=n_envs
        )
        
        current_model = model_path
        results.append({
            'stage': i + 1,
            'scenario': scenario,
            'model_path': model_path,
            **test_result
        })
        
        # 检查是否通过当前阶段
        if test_result['collision']:
            print(f"\n[WARNING] Stage {i+1} failed (collision). Consider retraining.")
    
    # 保存总结到文件
    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("CURRICULUM LEARNING SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Output Directory: {run_dir}\n")
        f.write(f"Start Stage: {start_stage}\n")
        f.write(f"Base Model: {base_model or 'None'}\n\n")
        f.write("Results:\n")
        for r in results:
            status = "PASS" if not r['collision'] else "FAIL"
            f.write(f"  Stage {r['stage']} ({r['scenario']}): x={r['final_x']:.1f}m, reward={r['total_reward']:.1f} [{status}]\n")
            f.write(f"    Model: {r['model_path']}\n")
    
    # 保存JSON格式结果
    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印总结
    print("\n" + "="*70)
    print("CURRICULUM COMPLETE")
    print(f"Output: {run_dir}")
    print("="*70)
    for r in results:
        status = "PASS" if not r['collision'] else "FAIL"
        print(f"  Stage {r['stage']} ({r['scenario']}): x={r['final_x']:.1f}m, reward={r['total_reward']:.1f} [{status}]")
    
    return run_dir, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum Learning Training")
    parser.add_argument("--start", type=int, default=0, help="Start from stage (0-indexed)")
    parser.add_argument("--base_model", type=str, default="models/pretrain_bc", 
                        help="Base model to continue from (default: pretrain_bc)")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # 检查预训练模型是否存在
    if args.base_model:
        # 支持带.zip和不带.zip两种格式
        model_path = args.base_model
        if model_path.endswith('.zip'):
            model_path = model_path[:-4]  # 去掉.zip
        if not os.path.exists(model_path + ".zip"):
            print(f"[WARNING] Base model not found: {model_path}.zip")
            print("Please run pretrain_bc.py first to create the pretrained model.")
            print("Or use --base_model None to start from scratch.")
            sys.exit(1)
        args.base_model = model_path  # 统一为不带.zip的格式
    
    run_curriculum(
        start_stage=args.start,
        base_model=args.base_model,
        n_envs=args.n_envs,
        output_dir=args.output
    )
