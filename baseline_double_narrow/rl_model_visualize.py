"""
使用训练好的RL模型进行窄门通行可视化 (RL Model Visualization)

基于curriculum_20260127_150853的最佳训练模型，生成与rule_based_double_narrow.py
相同格式的可视化结果（轨迹图+PDF输出）
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 设置字体为Arial
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import Rectangle


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import VehicleParams, Obstacle, OBSTACLE_SCENARIOS
from formation_rl_env import FormationRLEnv

# 导入stable_baselines3
try:
    from stable_baselines3 import PPO
except ImportError:
    print("ERROR: stable_baselines3 not installed. Install with: pip install stable-baselines3")
    sys.exit(1)


def load_model_from_path(model_path: str) -> str:
    """
    从完整路径加载模型
    
    参数:
        model_path: 模型文件的完整路径 (相对于PROJECT_ROOT或绝对路径)
    
    返回:
        完整的模型路径
    """
    # 如果是相对路径，相对于PROJECT_ROOT
    if not os.path.isabs(model_path):
        full_path = os.path.join(PROJECT_ROOT, model_path)
    else:
        full_path = model_path
    
    if not os.path.exists(full_path):
        print(f"❌ 模型文件不存在: {full_path}")
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"✅ 加载模型: {full_path}")
    return full_path


def run_rl_gate_test(
    model_path: str,
    scenario_name: str,
    max_steps: int = 2000,
    road_half_width: float = 2.5,
):
    """
    使用RL模型进行场景测试
    
    参数:
        model_path: 模型文件的完整路径
        scenario_name: 场景名称（从OBSTACLE_SCENARIOS中加载）
        max_steps: 最大仿真步数
        road_half_width: 道路半宽
    """
    vehicle_params = VehicleParams()
    car_radius = vehicle_params.car_radius
    
    # 从OBSTACLE_SCENARIOS加载场景配置
    if scenario_name not in OBSTACLE_SCENARIOS:
        print(f"❌ 未知场景: {scenario_name}")
        print(f"可用场景: {list(OBSTACLE_SCENARIOS.keys())}")
        return False
    
    obstacles = OBSTACLE_SCENARIOS[scenario_name]
    
    if not obstacles:
        print(f"⚠️  警告: 场景 '{scenario_name}' 没有障碍物（空场景）")
    
    # 加载训练模型
    try:
        full_model_path = load_model_from_path(model_path)
        model = PPO.load(full_model_path)
        print(f"✅ 成功加载模型")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    # 创建RL环境（使用特定场景）
    env = FormationRLEnv(scenario="main", num_cars=4, max_steps=max_steps)
    env.obstacles = obstacles  # 覆盖默认障碍物
    
    print(f"\n{'='*60}")
    print(f"RL Model Test: {scenario_name}")
    print(f"障碍物数量: {len(obstacles)}")
    print(f"最大步数: {max_steps}")
    print(f"{'='*60}")
    
    # 开始模拟
    obs, info = env.reset()
    trajectories = [[] for _ in range(4)]
    collision_log = []
    rewards = []
    
    for step in range(max_steps):
        # RL模型预测动作
        action, _ = model.predict(obs, deterministic=True)
        
        # 环境执行
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        # 记录轨迹
        for i, car in enumerate(env.cars):
            trajectories[i].append((car.x, car.y))
        
        # 碰撞检测
        if info.get('collision', False):
            avg_x = np.mean([car.x for car in env.cars])
            collision_log.append((step, avg_x, info.get('collision_type', 'unknown')))
        
        # 检查终止条件
        if terminated or truncated:
            break
    
    # 结果统计
    final_x = np.mean([car.x for car in env.cars])
    total_reward = sum(rewards)
    success = len(collision_log) == 0 and final_x >= 35.0
    
    print(f"\n🎯 测试结果:")
    print(f"  终点X坐标: {final_x:.1f}m")
    print(f"  总奖励: {total_reward:.1f}")
    print(f"  总步数: {len(rewards)}")
    print(f"  碰撞次数: {len(collision_log)}")
    print(f"  测试结果: {'✅ PASS' if success else '❌ FAIL'}")
    
    if collision_log:
        for step, x, ctype in collision_log:
            print(f"    碰撞 @ step {step}, x={x:.1f}m, type={ctype}")
    
    # 绘图 - 使用紧凑的宽扁格式
    car_length = vehicle_params.car_length
    car_width = vehicle_params.car_width
    max_x = max(max(p[0] for p in t) for t in trajectories if t) + 3
    
    # 根据场景长度动态调整图形尺寸
    fig_width = max(10, min(16, max_x / 8))  # 宽度在10-16之间
    fig_height = 3.0  # 固定高度
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 道路边界（不再填充灰色背景）
    ax.axhline(y=road_half_width, color='black', linestyle='--', linewidth=2)
    ax.axhline(y=-road_half_width, color='black', linestyle='--', linewidth=2)

    # 障碍物
    for obs in obstacles:
        if obs.obs_type == "circle":
            circle = plt.Circle((obs.x, obs.y), obs.radius, 
                              facecolor='gray', alpha=0.7, edgecolor='black')
            ax.add_patch(circle)
        else:  # rect or car
            ax.add_patch(Rectangle((obs.x-obs.width/2, obs.y-obs.height/2), 
                                   obs.width, obs.height,
                                   facecolor='gray', alpha=0.7, edgecolor='black'))

    # 车辆轨迹颜色
    colors = ['red', 'blue', 'green', 'orange']
    
    # 为每个障碍物位置找到快照索引
    snapshot_positions = []
    if obstacles:
        # 对每个障碍物，找到车辆通过时的位置
        for obs in obstacles:
            obs_x = obs.x
            # 在轨迹中找到最接近障碍物x坐标的点
            if trajectories[0]:
                best_idx = None
                min_dist = float('inf')
                for idx, (x, y) in enumerate(trajectories[0]):
                    dist = abs(x - obs_x)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx
                if best_idx is not None and min_dist < 5.0:  # 只有在5m范围内才显示
                    snapshot_positions.append(best_idx)
    
    # 如果没有障碍物或找不到快照位置，在起点和终点显示
    if not snapshot_positions:
        if trajectories[0]:
            snapshot_positions = [0, len(trajectories[0]) - 1]
    
    # 绘制轨迹线
    for i, traj in enumerate(trajectories):
        if traj:
            xs, ys = zip(*traj)
            # 轨迹线（虚线）
            ax.plot(xs, ys, color=colors[i], linestyle='--', 
                    label=f'UGV {i}', linewidth=1.5, alpha=0.7)
    
    # 绘制车辆快照（在每个障碍物位置）
    for snapshot_idx in snapshot_positions:
        for i, traj in enumerate(trajectories):
            if traj and snapshot_idx < len(traj):
                sx, sy = traj[snapshot_idx]
                # 使用实心矩形表示车辆
                ax.add_patch(Rectangle((sx-car_length/2, sy-car_width/2), 
                                       car_length, car_width,
                                       facecolor=colors[i], alpha=0.8, 
                                       edgecolor='black', linewidth=1.2, zorder=6))
    
    # 起点和终点标记（小方块）
    for i, traj in enumerate(trajectories):
        if traj:
            xs, ys = zip(*traj)
            # 起点标记（小方块）
            ax.plot(xs[0], ys[0], marker='s', markersize=4, 
                   color=colors[i], markeredgecolor='black', markeredgewidth=0.5, zorder=7)
            # 终点标记（小方块）
            ax.plot(xs[-1], ys[-1], marker='s', markersize=4, 
                   color=colors[i], markeredgecolor='black', markeredgewidth=0.5, zorder=7)

    # 场景名称映射（用于标题显示）
    scenario_title_map = {
        "main": "Main Test Scenario",
        "s1_right": "Right Obstacles",
        "s2_left": "Left Obstacles",
        "s1s2_mixed": "Mixed (Right & Left)",
        "s3_narrow": "Narrow Passage",
        "s4_center_small": "Center Small Obstacles",
        "s5_center_large": "Center Large Obstacles",
        "s1_s2_s5_mixed": "Mixed (Right & Left & Center Large)",
        "s6_very_narrow": "Very Narrow Passage",
        "double_narrow": "Double Narrow Gates",
        "empty": "Empty Scenario"
    }
    
    # 获取标题
    title = scenario_title_map.get(scenario_name, scenario_name)
    
    # 图形设置
    ax.set_xlim(0, max_x + 1)
    ax.set_ylim(-road_half_width - 0.5, road_half_width + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_title(title, fontsize=11)
    # 图例放在图下方，水平排列（位置下移避免遮挡X轴）
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.55), ncol=4, fontsize=9, frameon=True)
    ax.grid(True, alpha=0.35)

    # 保存图像
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.55)  # 为底部图例留出更多空间
    output_dir = CURRENT_DIR
    
    # 根据场景名称生成文件名
    base_filename = f"{scenario_name}"
    
    pdf_path = os.path.join(output_dir, f'{base_filename}.pdf')
    png_path = os.path.join(output_dir, f'{base_filename}.png')
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    
    print(f"\n📁 文件已保存:")
    print(f"  - {pdf_path}")
    print(f"  - {png_path}")
    
    plt.close(fig)

    env.close()
    return success


if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='使用训练好的RL模型进行场景可视化测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用方式:
  python rl_model_visualize.py <模型路径> <场景名称> [步数]

参数说明:
  模型路径: 模型文件的相对路径（相对于项目根目录）
  场景名称: 从OBSTACLE_SCENARIOS中选择的场景名称
  步数: 最大仿真步数（可选，默认2000）

可用场景:
  - main: 主测试场景（5种障碍物类型）
  - s1_right: 右侧障碍x3
  - s2_left: 左侧障碍x3
  - s1s2_mixed: 左右交替障碍
  - s3_narrow: 窄道x3（1.6m通道）
  - s4_center_small: 中间小障碍x3
  - s5_center_large: 中间大障碍x2
  - s1_s2_s5_mixed: 综合避障
  - s6_very_narrow: 极窄通道x2（1.0m通道）
  - double_narrow: 双窄门测试
  - empty: 空场景

示例:
  # 测试s1_s2_s5_mixed场景
  python rl_model_visualize.py "outputs/curriculum_20260128_210310/stage4_s1_s2_s5_mixed/best_model.zip" s1_s2_s5_mixed 2000
  
  # 测试s3_narrow场景，使用默认步数
  python rl_model_visualize.py "outputs/curriculum_20260127_150853/stage5_s3_narrow/best_model.zip" s3_narrow
  
  # 测试main场景（完整测试）
  python rl_model_visualize.py "outputs/curriculum_20260127_150853/stage5_s3_narrow/best_model.zip" main 3000

辅助命令:
  python rl_model_visualize.py --list-runs      # 列出所有可用的训练运行
  python rl_model_visualize.py --list-scenarios # 列出所有可用的场景
        """
    )
    
    # 位置参数
    parser.add_argument(
        'model_path',
        nargs='?',
        type=str,
        help='模型文件路径 (如: outputs/curriculum_20260128_210310/stage4_s1_s2_s5_mixed/best_model.zip)'
    )
    
    parser.add_argument(
        'scenario',
        nargs='?',
        type=str,
        help='场景名称 (如: s1_s2_s5_mixed, s3_narrow, main等)'
    )
    
    parser.add_argument(
        'steps',
        nargs='?',
        type=int,
        default=2000,
        help='最大步数 (默认: 2000)'
    )
    
    # 辅助功能
    parser.add_argument(
        '--list-runs',
        action='store_true',
        help='列出所有可用的训练运行目录'
    )
    
    parser.add_argument(
        '--list-scenarios',
        action='store_true',
        help='列出所有可用的场景'
    )
    
    args = parser.parse_args()
    
    # 列出训练运行
    if args.list_runs:
        outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
        if os.path.exists(outputs_dir):
            available_runs = [d for d in os.listdir(outputs_dir) 
                            if os.path.isdir(os.path.join(outputs_dir, d)) and d.startswith("curriculum_")]
            if available_runs:
                print("\n" + "="*60)
                print("可用的训练运行目录:")
                print("="*60)
                for i, run in enumerate(sorted(available_runs, reverse=True), 1):
                    run_path = os.path.join(outputs_dir, run)
                    # 列出该运行下的阶段
                    stages = [d for d in os.listdir(run_path) 
                             if os.path.isdir(os.path.join(run_path, d)) and d.startswith("stage")]
                    print(f"{i}. {run}")
                    if stages:
                        print(f"   阶段: {', '.join(sorted(stages))}")
                print("="*60)
            else:
                print("未找到训练运行目录")
        else:
            print(f"输出目录不存在: {outputs_dir}")
        sys.exit(0)
    
    # 列出可用场景
    if args.list_scenarios:
        print("\n" + "="*60)
        print("可用的场景:")
        print("="*60)
        for i, (name, obstacles) in enumerate(OBSTACLE_SCENARIOS.items(), 1):
            print(f"{i}. {name:20s} - {len(obstacles)}个障碍物")
        print("="*60)
        sys.exit(0)
    
    # 检查必要参数
    if not args.model_path:
        print("\n❌ 错误: 必须指定模型文件路径")
        print("\n使用方式:")
        print("  python rl_model_visualize.py <模型路径> <场景名称> [步数]")
        print("\n示例:")
        print('  python rl_model_visualize.py "outputs/curriculum_20260128_210310/stage4_s1_s2_s5_mixed/best_model.zip" s1_s2_s5_mixed 2000')
        print("\n或使用:")
        print("  --list-runs      查看可用的训练运行")
        print("  --list-scenarios 查看可用的场景")
        sys.exit(1)
    
    if not args.scenario:
        print("\n❌ 错误: 必须指定场景名称")
        print("\n可用场景:")
        for name in OBSTACLE_SCENARIOS.keys():
            print(f"  - {name}")
        print("\n示例:")
        print('  python rl_model_visualize.py "outputs/curriculum_20260128_210310/stage4_s1_s2_s5_mixed/best_model.zip" s1_s2_s5_mixed 2000')
        sys.exit(1)
    
    print("\n" + "="*60)
    print("RL模型可视化测试配置")
    print("="*60)
    print(f"模型路径: {args.model_path}")
    print(f"场景名称: {args.scenario}")
    print(f"最大步数: {args.steps}")
    print("="*60)
    
    # 执行测试
    success = run_rl_gate_test(
        model_path=args.model_path,
        scenario_name=args.scenario,
        max_steps=args.steps
    )
    
    # 输出结果
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    print(f"场景: {args.scenario}")
    print(f"结果: {'✅ PASS' if success else '❌ FAIL'}")
    print("="*60)
    print(f"\n📁 输出文件保存至: {CURRENT_DIR}")
    print(f"  - {args.scenario}.pdf")
    print(f"  - {args.scenario}.png")
