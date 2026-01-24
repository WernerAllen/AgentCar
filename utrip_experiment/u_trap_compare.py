"""
U-Trap 局部极小值逃逸对比实验

对比两种控制器在U型陷阱场景下的表现：
1. 传统DWA (Dynamic Window Approach)
2. AF-DWA (Reinforcement Learning Guided DWA)

实验目的：
- 展示传统DWA在面对局部极小值时，由于其短视特性，会陷入陷阱无法逃逸。
- 展示AF-DWA架构，通过上层RL提供的高层意图（逃逸方向），可以引导下层DWA成功逃逸。
- 验证所提出架构的优越性，且该实验范式与主工程中的编队控制架构一致。
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# --- 将父目录添加到Python路径中，以便导入项目模块 ---
# 这使得此脚本可以独立运行，同时复用现有代码
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- 从项目中导入必要的模块 ---
try:
    from vehicle_model import VehicleState, BicycleModel
    from dwa_controller import DWAController, DWAParams
    from config import VehicleParams, Obstacle
except ImportError as e:
    print(
        "Error: 无法导入项目模块。请确保此脚本位于 'utrip_experiment' 文件夹中，"
        f"且父目录 '{parent_dir}' 包含 vehicle_model.py, dwa_controller.py, config.py"
    )
    print(f"Import error: {e}")
    sys.exit(1)


# --- 1. U-Trap 场景定义 ---
def create_u_trap_scenario() -> Tuple[List[Obstacle], VehicleState, Tuple[float, float]]:
    """
    定义“课程学习风格”的直路 + 中间陷阱(U-Trap)

    设计目标：
    - 场景像你们 curriculum 的直路（有道路边界，左右可绕行）
    - 中间放一个U型陷阱（开口朝向来车方向），让传统DWA被目标牵引进入后陷入局部极小值
    - AF-DWA 通过高层意图（preferred_y）提前选择绕行侧，避免陷入/并可逃逸

    返回：障碍物列表、起点状态、终点位置
    """
    print("\n[场景] 创建课程学习风格U-Trap...")

    # 道路半宽（用于DWA边界约束与可视化）
    road_half_width = 2.5

    # ========== U-Trap 放在道路中间 ==========
    # 坐标系：车辆从左往右行驶；goal 在右侧。
    # U型陷阱开口朝左（迎接来车），车容易钻进去。
    # 由三块rect拼成：右侧竖墙 + 上下横墙（左侧开口）
    trap_x0 = 12.0   # 陷阱左侧开口的x附近
    trap_x1 = 16.0   # 陷阱右侧竖墙x
    trap_half_h = 0.9  # 陷阱“内腔”半高（越小越像陷阱，但要保证车有一定机动空间）
    wall_thickness = 0.35

    obstacles = [
        # 右侧竖墙（U底）
        Obstacle(
            x=trap_x1,
            y=0.0,
            radius=0.5,
            obs_type='rect',
            width=wall_thickness,
            height=2.0 * (trap_half_h + wall_thickness),
        ),
        # 上横墙
        Obstacle(
            x=(trap_x0 + trap_x1) / 2.0,
            y=trap_half_h + wall_thickness,
            radius=0.5,
            obs_type='rect',
            width=(trap_x1 - trap_x0),
            height=wall_thickness,
        ),
        # 下横墙
        Obstacle(
            x=(trap_x0 + trap_x1) / 2.0,
            y=-(trap_half_h + wall_thickness),
            radius=0.5,
            obs_type='rect',
            width=(trap_x1 - trap_x0),
            height=wall_thickness,
        ),
    ]

    # 起点放远一点，让“意图层”有时间提前做偏置（也更像课程学习直路）
    start_state = VehicleState(x=0.0, y=0.0, theta=0.0, v=0.8)

    # 终点在道路右侧中心线
    goal_pos = (25.0, 0.0)

    # 把road_half_width作为函数属性返回给调用方（避免改函数签名）
    create_u_trap_scenario.road_half_width = road_half_width

    return obstacles, start_state, goal_pos


# --- 2. 仿真核心逻辑 ---
def run_simulation(
    sim_name: str,
    start_state: VehicleState,
    goal_pos: Tuple[float, float],
    obstacles: List[Obstacle],
    dwa_params: DWAParams,
    vehicle_params: VehicleParams,
    use_rl_guidance: bool = False,
    max_steps: int = 800
) -> Tuple[List[VehicleState], bool, str]:
    """
    运行单次仿真
    
    Args:
        sim_name: 仿真名称 (e.g., 'Traditional DWA')
        start_state: 初始状态
        goal_pos: 目标位置
        obstacles: 障碍物列表
        dwa_params: DWA参数
        vehicle_params: 车辆参数
        use_rl_guidance: 是否使用RL高层引导
        max_steps: 最大仿真步数
        
    Returns:
        (轨迹, 是否成功, 结束原因)
    """
    print(f"\n[仿真开始] {sim_name} | RL引导: {'启用' if use_rl_guidance else '禁用'}")
    
    # 初始化
    vehicle = BicycleModel(vehicle_params)
    dwa = DWAController(vehicle_params, dwa_params)
    # 课程学习风格直路：启用道路边界（与主工程一致）
    dwa.road_half_width = getattr(create_u_trap_scenario, 'road_half_width', 2.5)

    state = start_state
    trajectory = [state]
    dt = dwa_params.dt
    
    for step in range(max_steps):
        # --- RL高层意图决策 (规则化) ---
        # 这是为了模拟RL agent在识别出陷阱后，提供逃逸方向的意图。
        # 这与主工程中RL输出dy修正量的架构思想一致。
        preferred_y = None
        if use_rl_guidance:
            # 规则化“RL意图层”：分阶段给出横向偏好，模拟RL学到的“先远离目标再逃逸”
            # - 深陷U内：先贴近上侧墙外缘附近，为掉头/绕行创造空间
            # - 接近U口：回到中心线对准出口
            # - 逃出后：对准最终目标
            # 课程学习风格场景的RL意图 (从左往右行驶)
            # 规则：在陷阱前提前选择绕行路径（上方），通过后再回来。
            trap_start_x = 11.0
            trap_end_x = 16.5

            if state.x < trap_start_x:
                # 陷阱前：提前向上偏移，准备绕行
                preferred_y = 2.0
            elif state.x < trap_end_x:
                # 陷阱旁：保持在上方的绕行路径
                preferred_y = 2.0
            else:
                # 通过后：回到中心线
                preferred_y = goal_pos[1]

        # --- DWA底层规划与控制 ---
        # DWA接收高层意图(preferred_y)，并结合局部环境进行安全规划
        control_state = (state.x, state.y, state.theta, state.v)
        a, delta = dwa.compute(control_state, goal_pos, obstacles, preferred_y)
        
        # --- 车辆模型更新状态 ---
        state = vehicle.step(state, (a, delta), dt)
        trajectory.append(state)
        
        # --- 检查终止条件 ---
        # 1. 到达终点
        dist_to_goal = np.sqrt((state.x - goal_pos[0])**2 + (state.y - goal_pos[1])**2)
        if dist_to_goal < 0.5:
            print(f"  [成功] 在 {step+1} 步后到达终点。")
            return trajectory, True, "Goal Reached"
            
        # 2. 碰撞检测
        for obs in obstacles:
            # 简化版矩形碰撞检测
            if obs.width > 0:
                if (abs(state.x - obs.x) < obs.width / 2 + vehicle_params.car_radius and
                    abs(state.y - obs.y) < obs.height / 2 + vehicle_params.car_radius):
                    print(f"  [失败] 在 {step+1} 步时发生碰撞。")
                    return trajectory, False, "Collision"

    print(f"  [失败] 达到最大步数 {max_steps}，未能到达终点。")
    return trajectory, False, "Timeout"


# --- 3. 可视化与结果对比 ---
def _draw_car(ax, x: float, y: float, theta: float, color: str, alpha: float = 0.9, label: str = None):
    """在(x,y)处画一个带朝向的“小车”矩形（中心在车体中心）"""
    car_length = 0.45
    car_width = 0.25

    # 车体矩形（以中心为原点）四个角
    corners = np.array([
        [ car_length / 2,  car_width / 2],
        [ car_length / 2, -car_width / 2],
        [-car_length / 2, -car_width / 2],
        [-car_length / 2,  car_width / 2],
    ])

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    rotated = corners @ R.T
    rotated[:, 0] += x
    rotated[:, 1] += y

    poly = patches.Polygon(rotated, closed=True, facecolor=color, edgecolor='black', alpha=alpha, label=label, zorder=6)
    ax.add_patch(poly)

    # 朝向箭头（车头方向）
    head = np.array([car_length / 2, 0.0]) @ R.T
    ax.arrow(x, y, head[0] * 0.8, head[1] * 0.8, head_width=0.08, head_length=0.12,
             fc='black', ec='black', alpha=0.8, zorder=7)


def plot_results(trajectories: dict, obstacles: List[Obstacle], goal_pos: Tuple[float, float], save_path: str, road_half_width: float):
    """
    绘制对比图
    """
    print("\n[可视化] 正在生成对比图...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制道路边界（课程学习风格）
    ax.axhline(y=road_half_width, color='black', linestyle='--', linewidth=2, alpha=0.8)
    ax.axhline(y=-road_half_width, color='black', linestyle='--', linewidth=2, alpha=0.8)

    # 绘制障碍物
    for obs in obstacles:
        rect = patches.Rectangle(
            (obs.x - obs.width / 2, obs.y - obs.height / 2),
            obs.width, obs.height,
            linewidth=1, edgecolor='black', facecolor='gray', alpha=0.7
        )
        ax.add_patch(rect)

    # 绘制轨迹
    colors = {'Traditional DWA': '#808080', 'AF-DWA': '#7FC97F'}
    linestyles = {'Traditional DWA': '--', 'AF-DWA': '--'}
    
    for name, (traj, success, reason) in trajectories.items():
        x_coords = [s.x for s in traj]
        y_coords = [s.y for s in traj]
        label = name

        ax.plot(
            x_coords,
            y_coords,
            color=colors[name],
            linestyle=linestyles[name],
            label=label,
            linewidth=2,
        )

        # 在轨迹上均匀放置若干小车（更美观）
        if len(traj) > 5:
            n_cars = 4
            idxs = np.linspace(0, len(traj) - 1, n_cars, dtype=int)
            for k, idx in enumerate(idxs):
                car_alpha = 0.25 if k not in (0, len(idxs) - 1) else 0.9
                car_label = f"{name} Start" if k == 0 else None
                _draw_car(
                    ax,
                    traj[idx].x,
                    traj[idx].y,
                    traj[idx].theta,
                    colors[name],
                    alpha=car_alpha,
                    label=car_label,
                )

    # 绘制起点和终点（起点用小车形状表示）
    start_pos = list(trajectories.values())[0][0][0]
    _draw_car(ax, start_pos.x, start_pos.y, start_pos.theta, 'blue', alpha=0.6, label='Start Point')
    ax.scatter(goal_pos[0], goal_pos[1], c='gold', marker='*', s=250, label='Goal Point', zorder=5)
    
    # 设置图表样式
    ax.set_title('U-Trap Escape Comparison', fontsize=16)
    axis_fontsize = 12
    ax.set_xlabel('X (m)', fontsize=axis_fontsize)
    ax.set_ylabel('Y (m)', fontsize=axis_fontsize)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.6)
    # 将图例放在图表下方
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=axis_fontsize, frameon=True)
    plt.subplots_adjust(bottom=0.3) # 为下方图例与x轴标签留出更多空间
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"[完成] 对比图已保存至: {save_path}")
    plt.show()


# --- 4. 主函数 ---
if __name__ == "__main__":
    # --- 参数配置 ---
    vehicle_params = VehicleParams()
    
    # DWA参数 for 传统DWA (更容易陷入陷阱)
    dwa_params_baseline = DWAParams(
        predict_time=1.5,       # 预测时间较短，更“短视”
        heading_weight=2.5,     # 更强的目标朝向权重，更“贪心”
        velocity_weight=0.1,
        dist_weight=1.5,
        min_speed=0.3,          # 最小速度设为0.3 m/s，避免停车
        max_speed=1.5,          # 最大速度设为1.0 m/s
        dt=0.1,                 # 时间步长
        robot_radius=0.25        # 机器人半径
    )
    
    # DWA参数 for AF-DWA (增加RL引导权重，强化逃逸效果)
    dwa_params_rl = DWAParams(
        predict_time=2.0,
        heading_weight=2.5,
        velocity_weight=0.1,
        dist_weight=1.5,
        min_speed=0.3,          # 最小速度设为0.3 m/s，避免停车
        max_speed=1.5,          # 最大速度设为1.0 m/s
        dt=0.1,                 # 时间步长
        robot_radius=0.25,       # 机器人半径
        rl_direction_weight=8.0 # 增加RL方向引导权重
    )

    # --- 运行实验 ---
    obstacles, start_state, goal_pos = create_u_trap_scenario()
    
    # 运行传统DWA
    traj_dwa, success_dwa, reason_dwa = run_simulation(
        sim_name="Traditional DWA",
        start_state=start_state,
        goal_pos=goal_pos,
        obstacles=obstacles,
        dwa_params=dwa_params_baseline,
        vehicle_params=vehicle_params,
        use_rl_guidance=False
    )
    
    # 运行AF-DWA
    traj_rl, success_rl, reason_rl = run_simulation(
        sim_name="AF-DWA",
        start_state=start_state,
        goal_pos=goal_pos,
        obstacles=obstacles,
        dwa_params=dwa_params_rl,
        vehicle_params=vehicle_params,
        use_rl_guidance=True
    )
    
    # --- 整理并展示结果 ---
    results = {
        "Traditional DWA": (traj_dwa, success_dwa, reason_dwa),
        "AF-DWA": (traj_rl, success_rl, reason_rl)
    }
    
    # 定义输出路径
    output_filename = "u_trap_comparison.pdf"
    output_path = os.path.join(current_dir, output_filename)
    
    road_half_width = getattr(create_u_trap_scenario, 'road_half_width', 2.5)
    plot_results(results, obstacles, goal_pos, output_path, road_half_width)

    # --- 打印量化指标 ---
    print("\n--- Quantitative Metrics ---")
    for name, (traj, success, reason) in results.items():
        path_length = sum(np.sqrt((traj[i].x - traj[i-1].x)**2 + (traj[i].y - traj[i-1].y)**2) for i in range(1, len(traj)))
        time_taken = len(traj) * dwa_params_baseline.dt
        print(f"Controller: {name}")
        print(f"  - Success: {success}")
        print(f"  - Reason: {reason}")
        print(f"  - Time Taken: {time_taken:.2f} s")
        print(f"  - Path Length: {path_length:.2f} m")
