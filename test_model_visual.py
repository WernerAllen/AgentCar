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
    
    # 绘制轨迹图（上方轨迹 + 下方编队位置图）
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(24, 10), height_ratios=[1, 0.5],
                                   gridspec_kw={'hspace': 0.25})
    
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
    
    # ===== 下方子图：编队快照图 =====
    # 显示关键阶段的编队位置（减少到5个，更清晰）
    snapshot_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    snapshot_labels = ['Start', '25%', '50%', '75%', 'End']
    
    # 每个快照之间的间距
    spacing = 6
    
    for snap_idx, (ratio, label) in enumerate(zip(snapshot_ratios, snapshot_labels)):
        offset_x = snap_idx * spacing + 2
        
        # 收集该阶段所有车的位置
        snap_positions = []
        for i, traj in enumerate(trajectories):
            if traj:
                idx = int(len(traj) * ratio) if ratio < 1.0 else len(traj) - 1
                if 0 <= idx < len(traj):
                    snap_positions.append((i, traj[idx][0], traj[idx][1]))
        
        if snap_positions:
            center_x = np.mean([p[1] for p in snap_positions])
            center_y = np.mean([p[2] for p in snap_positions])
            
            # 计算原始编队范围
            orig_xs = [(p[1] - center_x) for p in snap_positions]
            orig_ys = [(p[2] - center_y) for p in snap_positions]
            
            # 计算缩放比例，确保编队适应区域（最大宽度spacing-1，最大高度road_hw*1.5）
            max_width = spacing - 2
            max_height = road_hw * 1.5
            x_range = max(orig_xs) - min(orig_xs) + car_length if orig_xs else 1
            y_range = max(orig_ys) - min(orig_ys) + car_width if orig_ys else 1
            scale = min(max_width / max(x_range, 0.1), max_height / max(y_range, 0.1), 1.5)
            
            all_rel_x = []
            all_y = []
            
            for i, orig_x, orig_y in snap_positions:
                rel_x = (orig_x - center_x) * scale
                rel_y = center_y + (orig_y - center_y) * min(scale, 1.0)  # Y方向保持或缩小
                all_rel_x.append(offset_x + rel_x)
                all_y.append(rel_y)
                car_rect = Rectangle(
                    (offset_x + rel_x - car_length/2, rel_y - car_width/2),
                    car_length, car_width,
                    facecolor=colors[i], alpha=0.8, edgecolor='black', linewidth=1.5, zorder=5
                )
                ax2.add_patch(car_rect)
            
            # 用线条连接所有车形成外围轮廓
            from scipy.spatial import ConvexHull
            points = np.array(list(zip(all_rel_x, all_y)))
            if len(points) >= 3:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])  # 闭合
                ax2.plot(hull_points[:, 0], hull_points[:, 1], 
                        color='purple', linewidth=2, linestyle='-', alpha=0.7, zorder=3)
        
        # 添加阶段标签
        ax2.text(offset_x, road_hw + 0.5, label, ha='center', fontsize=11, fontweight='bold')
        # 添加分隔线
        if snap_idx < len(snapshot_ratios) - 1:
            ax2.axvline(x=offset_x + spacing/2 + 1, color='lightgray', linestyle='-', linewidth=1, alpha=0.5)
    
    # 设置坐标轴
    ax2.set_xlim(-1, len(snapshot_ratios) * spacing + 2)
    ax2.set_ylim(-road_hw - 1, road_hw + 1)
    ax2.set_ylabel('Y (m)', fontsize=14)
    ax2.set_title('Formation Snapshots at Different Stages', fontsize=14)
    ax2.set_xticks([])
    ax2.grid(True, alpha=0.3, axis='y')
    
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
