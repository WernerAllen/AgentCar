"""
可视化测试PPO模型
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from stable_baselines3 import PPO
from formation_rl_env import FormationRLEnv


def test_model_visual(model_path: str, scenario: str = "s1_right", max_steps: int = 2500):
    """可视化测试模型"""
    
    print(f"\n{'='*60}")
    print(f"Visual Test: {scenario}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 创建环境
    env = FormationRLEnv(scenario=scenario, num_cars=4, max_steps=max_steps)
    obs, info = env.reset()
    
    # 记录轨迹
    trajectories = [[] for _ in range(4)]
    rewards = []
    
    done = False
    step = 0
    total_reward = 0
    
    while not done and step < max_steps:
        # 使用模型预测动作
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 记录位置
        for i, car in enumerate(env.cars):
            trajectories[i].append((car.x, car.y))
        
        rewards.append(reward)
        total_reward += reward
        done = terminated or truncated
        step += 1
        
        # 每100步打印进度
        if step % 100 == 0:
            avg_x = np.mean([car.x for car in env.cars])
            print(f"Step {step}: avg_x={avg_x:.1f}m, reward={reward:.2f}")
    
    # 最终状态
    final_x = np.mean([car.x for car in env.cars])
    car_ys = [car.y for car in env.cars]
    
    print(f"\n--- Result ---")
    print(f"Steps: {step}")
    print(f"Final X: {final_x:.1f}m")
    print(f"Y range: [{min(car_ys):.2f}, {max(car_ys):.2f}]")
    print(f"Total Reward: {total_reward:.1f}")
    print(f"Status: {'Collision' if terminated else 'Success/Truncated'}")
    
    # 绘制轨迹图（单图，更长比例）
    fig, ax = plt.subplots(1, 1, figsize=(24, 6))
    
    # 小车实际尺寸
    car_length = 0.45
    car_width = 0.25
    
    # 轨迹图：使用不同线型避免重叠遮挡
    colors = ['red', 'blue', 'green', 'orange']
    linestyles = ['-', '--', '-.', ':']
    
    for i, traj in enumerate(trajectories):
        if traj:
            xs, ys = zip(*traj)
            ax.plot(xs, ys, color=colors[i], linestyle=linestyles[i], 
                    label=f'Car {i}', linewidth=1.2, alpha=0.85)  # 更细的轨迹线
            
            # 绘制起点小车轮廓（矩形，实线边框）
            start_rect = Rectangle(
                (xs[0] - car_length/2, ys[0] - car_width/2),
                car_length, car_width, 
                facecolor=colors[i], alpha=0.8, edgecolor='black', linewidth=1.5, zorder=5
            )
            ax.add_patch(start_rect)
            
            # 绘制终点小车轮廓（矩形，虚线边框）
            end_rect = Rectangle(
                (xs[-1] - car_length/2, ys[-1] - car_width/2),
                car_length, car_width,
                facecolor=colors[i], alpha=0.5, edgecolor='black', linewidth=1.5, 
                linestyle='--', zorder=5
            )
            ax.add_patch(end_rect)
            
            # ===== 中间快照机制 =====
            # 在轨迹的多个位置绘制小车矩形轮廓
            snapshot_ratios = [0.25, 0.5, 0.75]
            for ratio in snapshot_ratios:
                idx = int(len(xs) * ratio)
                if 0 < idx < len(xs) - 1:
                    # 使用小车矩形轮廓（与起点/终点一致）
                    snap_rect = Rectangle(
                        (xs[idx] - car_length/2, ys[idx] - car_width/2),
                        car_length, car_width,
                        facecolor=colors[i], alpha=0.6, edgecolor='black', 
                        linewidth=1, zorder=6
                    )
                    ax.add_patch(snap_rect)
    
    # 绘制起点编队框（虚线矩形标注2x2编队）
    start_xs = [traj[0][0] for traj in trajectories if traj]
    start_ys = [traj[0][1] for traj in trajectories if traj]
    formation_rect = Rectangle(
        (min(start_xs) - 0.5, min(start_ys) - 0.5),
        max(start_xs) - min(start_xs) + 1,
        max(start_ys) - min(start_ys) + 1,
        fill=False, edgecolor='purple', linewidth=2, linestyle=':', zorder=4,
        label='Formation'
    )
    ax.add_patch(formation_rect)
    
    # ===== 中间快照编队框 + 垂直标识线 =====
    # 在快照位置绘制垂直虚线和编队框，使其在长轨迹上也清晰可见
    snapshot_ratios = [0.25, 0.5, 0.75]
    snapshot_colors = ['green', 'orange', 'red']  # 25%绿 50%橙 75%红
    for ratio, snap_color in zip(snapshot_ratios, snapshot_colors):
        snap_xs = []
        snap_ys = []
        for traj in trajectories:
            if traj:
                idx = int(len(traj) * ratio)
                if 0 < idx < len(traj):
                    snap_xs.append(traj[idx][0])
                    snap_ys.append(traj[idx][1])
        if snap_xs and snap_ys:
            # 垂直虚线标识快照位置
            mean_x = np.mean(snap_xs)
            ax.axvline(x=mean_x, color=snap_color, linestyle=':', 
                      linewidth=1.5, alpha=0.6, zorder=2,
                      label=f'{int(ratio*100)}% snapshot' if ratio == 0.5 else None)
            # 编队框
            snap_formation_rect = Rectangle(
                (min(snap_xs) - 0.3, min(snap_ys) - 0.3),
                max(snap_xs) - min(snap_xs) + 0.6,
                max(snap_ys) - min(snap_ys) + 0.6,
                fill=False, edgecolor=snap_color, linewidth=2.5, linestyle='--', 
                alpha=0.9, zorder=6
            )
            ax.add_patch(snap_formation_rect)
    
    # 绘制障碍物
    for obs in env.obstacles:
        if obs.width > 0 and obs.height > 0:
            rect = Rectangle(
                (obs.x - obs.width/2, obs.y - obs.height/2),
                obs.width, obs.height,
                color='gray', alpha=0.7
            )
            ax.add_patch(rect)
        else:
            circle = Circle((obs.x, obs.y), obs.radius, color='gray', alpha=0.7)
            ax.add_patch(circle)
    
    # 绘制道路边界
    road_hw = env.env_params.road_half_width
    ax.axhline(y=road_hw, color='black', linestyle='--', linewidth=2)
    ax.axhline(y=-road_hw, color='black', linestyle='--', linewidth=2)
    
    # 动态设置x轴范围
    max_x = max(max(xs) for traj in trajectories if traj for xs, ys in [zip(*traj)])
    ax.set_xlim(-2, max_x + 10)
    ax.set_ylim(-road_hw - 1.0, road_hw + 1.0)  # 稍微增大Y轴范围
    ax.set_aspect('equal')  # 保持真实比例
    
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.set_title(f'Trajectory - {scenario} (Total Reward: {total_reward:.1f})', fontsize=16)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存到test_results文件夹
    import os
    save_dir = os.path.join(os.path.dirname(__file__), 'test_results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'test_result_{scenario}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.show()
    
    env.close()
    return total_reward, step, terminated


if __name__ == "__main__":
    import sys
    
    # 默认使用最新的best_model
    model_path = "models/ppo_formation_20260117_174253/best_model.zip"
    scenario = "s1_right"
    max_steps = 500
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        scenario = sys.argv[2]
    if len(sys.argv) > 3:
        max_steps = int(sys.argv[3])
    
    # main场景需要更多步数
    if scenario == "main" and max_steps < 2000:
        max_steps = 2000
    
    test_model_visual(model_path, scenario, max_steps)
