import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# Ensure project root is on path so we can reuse existing dynamics/DWA/Obstacle definitions
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import VehicleParams, Obstacle
from vehicle_model import VehicleState, BicycleModel
from dwa_controller import DWAController, DWAParams


class Mode(str, Enum):
    MODE_MAINTAIN = "MODE_MAINTAIN"
    MODE_RECONFIGURE = "MODE_RECONFIGURE"
    MODE_AVOID = "MODE_AVOID"


@dataclass
class Baseline2HyperParams:
    # Safe distance threshold
    D_safe: float = 1.2

    # How far ahead (x) to look when estimating gap
    gap_lookahead_x: float = 15.0

    # Gap computation: treat obstacles as blocking intervals expanded by vehicle radius
    gap_expand_margin: float = 0.0

    # Formation offsets (relative to virtual leader point) - 2x2矩形队形
    #   Car0  Car1
    #   Car2  Car3
    rect_offsets: Tuple[Tuple[float, float], ...] = (
        (1.0, 1.2),       # Car 0: 前左
        (1.0, -1.2),      # Car 1: 前右
        (-1.0, 1.2),      # Car 2: 后左
        (-1.0, -1.2),     # Car 3: 后右
    )

    # 一字长蛇阵 - 纵向单列排列 (所有车y=0)
    #   Car0 -> Car1 -> Car2 -> Car3
    line_offsets: Tuple[Tuple[float, float], ...] = (
        (2.0, 0.0),       # Car 0: 最前
        (0.0, 0.0),       # Car 1
        (-2.0, 0.0),      # Car 2
        (-4.0, 0.0),      # Car 3: 最后
    )

    # Controller gains (simple PD tracking to virtual structure points)
    kp_y: float = 2.5
    kd_y: float = 0.8

    # Convert y-error to steering
    steer_gain: float = 0.35

    # Target cruise speed
    v_ref: float = 1.2
    
    # 队形切换平滑过渡时间（秒）
    transition_time: float = 3.0


@dataclass
class SceneMetrics:
    D0: float
    d_gap: float
    H_form: float
    h_ugv: float


def rect_interval_for_obstacle(obs: Obstacle, y_expand: float) -> Optional[Tuple[float, float]]:
    if obs.width > 0 and obs.height > 0:
        half_h = obs.height / 2.0
        return (obs.y - half_h - y_expand, obs.y + half_h + y_expand)
    # circle-like
    return (obs.y - obs.radius - y_expand, obs.y + obs.radius + y_expand)


def min_distance_to_obstacles(state: VehicleState, obstacles: List[Obstacle], vehicle_params: VehicleParams) -> float:
    dmin = float("inf")
    for obs in obstacles:
        if obs.width > 0 and obs.height > 0:
            half_w = obs.width / 2.0
            half_h = obs.height / 2.0
            dx = max(0.0, abs(state.x - obs.x) - half_w)
            dy = max(0.0, abs(state.y - obs.y) - half_h)
            dist = float(np.sqrt(dx * dx + dy * dy))
        else:
            dist = float(np.sqrt((state.x - obs.x) ** 2 + (state.y - obs.y) ** 2) - obs.radius)
        dmin = min(dmin, dist)

    dmin = dmin - vehicle_params.car_radius
    return dmin


def estimate_gap_width(
    leader_x: float,
    obstacles: List[Obstacle],
    road_half_width: float,
    vehicle_params: VehicleParams,
    hp: Baseline2HyperParams,
) -> float:
    # Find obstacles ahead within [leader_x, leader_x + lookahead]
    candidates: List[Obstacle] = []
    for obs in obstacles:
        if obs.x >= leader_x and obs.x <= leader_x + hp.gap_lookahead_x:
            if obs.width > 0 or obs.radius > 0:
                candidates.append(obs)

    # If no obstacles ahead, gap is full road
    if not candidates:
        return 2.0 * road_half_width

    y_expand = vehicle_params.car_radius + hp.gap_expand_margin

    # Build blocked intervals in y
    intervals: List[Tuple[float, float]] = []
    for obs in candidates:
        intervals.append(rect_interval_for_obstacle(obs, y_expand))

    # Add road boundary as blocked outside [-road_half_width, road_half_width]
    # We compute free gaps inside the road, so no need to add boundary intervals.

    # Merge intervals
    intervals.sort(key=lambda t: t[0])
    merged: List[Tuple[float, float]] = []
    for a, b in intervals:
        if not merged:
            merged.append((a, b))
        else:
            la, lb = merged[-1]
            if a <= lb:
                merged[-1] = (la, max(lb, b))
            else:
                merged.append((a, b))

    # Clip merged intervals to road range
    clipped: List[Tuple[float, float]] = []
    for a, b in merged:
        a2 = max(a, -road_half_width)
        b2 = min(b, road_half_width)
        if a2 < b2:
            clipped.append((a2, b2))

    # Compute max free gap between blocked intervals
    free_max = 0.0
    cursor = -road_half_width
    for a, b in clipped:
        free_max = max(free_max, a - cursor)
        cursor = max(cursor, b)
    free_max = max(free_max, road_half_width - cursor)

    return float(max(0.0, free_max))


def formation_width(offsets: Tuple[Tuple[float, float], ...], vehicle_params: VehicleParams) -> float:
    ys = [o[1] for o in offsets]
    return float((max(ys) - min(ys)) + 2.0 * vehicle_params.car_radius)


def choose_mode(metrics: SceneMetrics, hp: Baseline2HyperParams) -> Tuple[Mode, str]:
    if metrics.D0 < hp.D_safe:
        return Mode.MODE_AVOID, "D0<D_safe"

    if metrics.d_gap > metrics.H_form:
        return Mode.MODE_MAINTAIN, "gap>H_form"

    if metrics.h_ugv < metrics.d_gap <= metrics.H_form:
        return Mode.MODE_RECONFIGURE, "h_ugv<gap<=H_form"

    return Mode.MODE_AVOID, "gap<=h_ugv"


def leader_control_to_goal(state: VehicleState, goal: Tuple[float, float], hp: Baseline2HyperParams) -> Tuple[float, float]:
    """领航者控制：简单的纯追踪风格控制"""
    dx = goal[0] - state.x
    dy = goal[1] - state.y
    
    # 目标方向角
    target_angle = np.arctan2(dy, dx)
    
    # 角度误差
    angle_err = target_angle - state.theta
    # 归一化到 [-pi, pi]
    while angle_err > np.pi:
        angle_err -= 2 * np.pi
    while angle_err < -np.pi:
        angle_err += 2 * np.pi
    
    # 转向控制
    delta = float(np.clip(1.0 * angle_err, -np.pi / 6, np.pi / 6))
    
    # 速度控制
    a = 1.5 * (hp.v_ref - state.v)
    
    return float(a), delta


def follower_virtual_structure_control(
    follower: VehicleState,
    leader: VehicleState,
    offset: Tuple[float, float],
    hp: Baseline2HyperParams,
) -> Tuple[float, float]:
    """
    跟随者控制：追踪虚拟结构点
    
    offset[0]: x方向偏移（负值表示在leader后方）
    offset[1]: y方向偏移
    """
    # 目标点（世界坐标系，假设leader朝向与x轴对齐）
    target_x = leader.x + offset[0]
    target_y = leader.y + offset[1]
    
    # 计算到目标点的误差
    dx = target_x - follower.x
    dy = target_y - follower.y
    dist = np.sqrt(dx * dx + dy * dy)
    
    # 目标方向角
    target_angle = np.arctan2(dy, dx)
    
    # 角度误差
    angle_err = target_angle - follower.theta
    while angle_err > np.pi:
        angle_err -= 2 * np.pi
    while angle_err < -np.pi:
        angle_err += 2 * np.pi
    
    # 转向控制 - 根据角度误差
    delta = float(np.clip(1.5 * angle_err, -np.pi / 6, np.pi / 6))
    
    # 速度控制 - 根据距离误差调整
    # 如果落后太多，加速；如果太近，减速
    target_v = hp.v_ref
    if dist > 0.5:  # 落后太多
        target_v = hp.v_ref * 1.3
    elif dist < 0.2:  # 太近
        target_v = hp.v_ref * 0.8
    
    a = float(2.0 * (target_v - follower.v))
    
    return a, delta


def interpolate_offsets(
    start_offsets: Tuple[Tuple[float, float], ...],
    end_offsets: Tuple[Tuple[float, float], ...],
    alpha: float,
) -> Tuple[Tuple[float, float], ...]:
    """在两个队形之间线性插值"""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    result = []
    for s, e in zip(start_offsets, end_offsets):
        x = s[0] * (1 - alpha) + e[0] * alpha
        y = s[1] * (1 - alpha) + e[1] * alpha
        result.append((x, y))
    return tuple(result)


def run_formation_switch_demo(
    out_dir: str,
    max_steps: int = 500,
    dt: float = 0.1,
    road_half_width: float = 2.5,
    narrow_zone_start: float = 25.0,
    narrow_zone_end: float = 35.0,
    lookahead_dist: float = 8.0,
):
    """
    空白场景队形切换演示
    
    变阵方法：通过速度差实现
    - RECT->LINE: y正的车加速，y负的车减速/停止，自然形成纵列
    - LINE->RECT: 反过来，y负的加速追上
    """
    os.makedirs(out_dir, exist_ok=True)

    vehicle_params = VehicleParams()
    hp = Baseline2HyperParams()

    obstacles: List[Obstacle] = []
    
    # 初始位置 (x, y) 和速度
    # 2x2矩形队形
    #   Car0(y=1.2)   Car1(y=-1.2)
    #   Car2(y=1.2)   Car3(y=-1.2)
    car_states = [
        {'x': 5.0, 'y': 1.2, 'v': hp.v_ref},   # Car 0: 前左 (y正)
        {'x': 5.0, 'y': -1.2, 'v': hp.v_ref},  # Car 1: 前右 (y负)
        {'x': 3.0, 'y': 1.2, 'v': hp.v_ref},   # Car 2: 后左 (y正)
        {'x': 3.0, 'y': -1.2, 'v': hp.v_ref},  # Car 3: 后右 (y负)
    ]

    goal_x = 55.0
    
    # 变阵时机
    switch_to_line_x = narrow_zone_start - lookahead_dist
    switch_to_rect_x = narrow_zone_end + 2.0

    # Logs
    traj = [[] for _ in range(4)]
    
    # 状态机: 0=RECT, 1=TO_LINE, 2=LINE, 3=TO_RECT, 4=RECT_FINAL
    current_mode = 0
    
    # 速度参数
    v_fast = hp.v_ref * 1.8   # 加速
    v_slow = hp.v_ref * 0.3   # 减速
    v_normal = hp.v_ref

    print("="*60)
    print("空白场景队形切换演示 (速度差变阵)")
    print(f"初始队形: 2x2矩形")
    print(f"狭窄区域: x={narrow_zone_start}m ~ {narrow_zone_end}m")
    print(f"变阵方法: y正的车加速，y负的车减速")
    print("="*60)

    for step in range(max_steps):
        # 获取当前编队中心x位置
        avg_x = sum(c['x'] for c in car_states) / 4
        
        # 决策逻辑 - 单向状态机
        if current_mode == 0:  # RECT
            if avg_x >= switch_to_line_x:
                current_mode = 1
                print(f"[Step {step}] x={avg_x:.1f}m: 开始变阵 RECT -> LINE")
        
        elif current_mode == 1:  # TO_LINE
            # 检查是否已经形成一字长蛇阵
            y_pos_min_x = min(car_states[0]['x'], car_states[2]['x'])
            y_neg_max_x = max(car_states[1]['x'], car_states[3]['x'])
            if y_pos_min_x > y_neg_max_x + 1.5:
                current_mode = 2
                print(f"[Step {step}] x={avg_x:.1f}m: 变阵完成 -> LINE")
        
        elif current_mode == 2:  # LINE
            if avg_x >= switch_to_rect_x:
                current_mode = 3
                print(f"[Step {step}] x={avg_x:.1f}m: 开始恢复 RECT")
        
        elif current_mode == 3:  # TO_RECT
            # 检查是否已经恢复矩形
            if abs(car_states[0]['x'] - car_states[1]['x']) < 0.5 and \
               abs(car_states[2]['x'] - car_states[3]['x']) < 0.5:
                current_mode = 4
                print(f"[Step {step}] x={avg_x:.1f}m: 恢复完成 -> RECT")
        
        # current_mode == 4: RECT_FINAL, 不再切换
        
        # 根据模式设置每辆车的速度和y目标
        for i, car in enumerate(car_states):
            is_y_positive = (i == 0 or i == 2)  # Car 0, 2 是 y正
            
            if current_mode == 0 or current_mode == 4:  # RECT
                car['v'] = v_normal
                target_y = 1.2 if is_y_positive else -1.2
            
            elif current_mode == 1:  # TO_LINE
                # y正的加速，y负的减速
                car['v'] = v_fast if is_y_positive else v_slow
                # y向中心收敛
                target_y = 0.2 if is_y_positive else -0.2
            
            elif current_mode == 2:  # LINE
                car['v'] = v_normal
                target_y = 0.0  # 一字排开
            
            elif current_mode == 3:  # TO_RECT
                # y负的加速追上，y正的减速等待
                car['v'] = v_slow if is_y_positive else v_fast
                target_y = 1.2 if is_y_positive else -1.2
            
            # 更新位置
            car['x'] += car['v'] * dt
            # y方向平滑辽近目标
            car['y'] += (target_y - car['y']) * 0.1

        # 记录轨迹
        for i, car in enumerate(car_states):
            traj[i].append((car['x'], car['y']))

        if avg_x >= goal_x:
            break

    # 绘图
    plot_formation_switch_result_v2(traj, obstacles, road_half_width, (goal_x, 0.0), 
                                     switch_to_line_x, narrow_zone_start, narrow_zone_end,
                                     switch_to_rect_x, out_dir, vehicle_params)
    
    print(f"\n演示完成！输出保存到: {out_dir}")


def plot_formation_switch_result_v2(
    traj: List[List[Tuple[float, float]]],
    obstacles: List[Obstacle],
    road_half_width: float,
    goal: Tuple[float, float],
    switch_to_line_x: float,
    narrow_zone_start: float,
    narrow_zone_end: float,
    switch_to_rect_x: float,
    out_dir: str,
    vehicle_params: VehicleParams,
):
    """绘制队形切换结果 - 参考test_model_visual.py风格"""
    from matplotlib.patches import Rectangle
    
    fig, ax = plt.subplots(figsize=(20, 5))
    
    car_length = vehicle_params.car_length
    car_width = vehicle_params.car_width
    
    # 道路填充
    max_x = max(max(p[0] for p in t) for t in traj if t) + 5
    ax.fill_between([0, max_x], [-road_half_width, -road_half_width], 
                    [road_half_width, road_half_width], 
                    color='lightgray', alpha=0.3)
    
    # 道路边界
    ax.axhline(y=road_half_width, color='black', linestyle='--', linewidth=2)
    ax.axhline(y=-road_half_width, color='black', linestyle='--', linewidth=2)
    
    # 狭窄区域（障碍物位置）- 红色标记
    ax.fill_betweenx([-road_half_width, road_half_width], narrow_zone_start, narrow_zone_end, 
                     alpha=0.2, color='red', label='Narrow Zone')
    ax.axvline(x=narrow_zone_start, color='red', linestyle='-', linewidth=2, alpha=0.8)
    ax.axvline(x=narrow_zone_end, color='red', linestyle='-', linewidth=2, alpha=0.8)
    
    # 变阵区域（一字长蛇阵区间）- 黄色标记
    ax.axvline(x=switch_to_line_x, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=switch_to_rect_x, color='green', linestyle='--', linewidth=2, alpha=0.7)
    
    # 区域标签
    ax.text((0 + switch_to_line_x)/2, road_half_width + 0.3, '2x2 Rect', 
            ha='center', fontsize=10, fontweight='bold', color='blue')
    ax.text((switch_to_line_x + narrow_zone_start)/2, road_half_width + 0.3, 'Transition', 
            ha='center', fontsize=9, color='orange')
    ax.text((narrow_zone_start + narrow_zone_end)/2, road_half_width + 0.3, '1x4 Line', 
            ha='center', fontsize=10, fontweight='bold', color='darkorange')
    ax.text((narrow_zone_end + switch_to_rect_x)/2, road_half_width + 0.3, 'Trans', 
            ha='center', fontsize=9, color='green')
    ax.text((switch_to_rect_x + max_x)/2, road_half_width + 0.3, '2x2 Rect', 
            ha='center', fontsize=10, fontweight='bold', color='blue')
    
    # 轨迹 - 使用不同线型区分
    colors = ['red', 'blue', 'green', 'orange']
    linestyles = ['-', '--', '-.', ':']
    
    for i, t in enumerate(traj):
        if t:
            xs, ys = zip(*t)
            ax.plot(xs, ys, color=colors[i], linestyle=linestyles[i], 
                    label=f'Car {i}', linewidth=1.5, alpha=0.85)
            
            # 起点小车轮廓（矩形，实线边框）
            start_rect = Rectangle(
                (xs[0] - car_length/2, ys[0] - car_width/2),
                car_length, car_width, 
                facecolor=colors[i], alpha=0.9, edgecolor='black', linewidth=1.5, zorder=5
            )
            ax.add_patch(start_rect)
            
            # 终点小车轮廓（矩形，虚线边框）
            end_rect = Rectangle(
                (xs[-1] - car_length/2, ys[-1] - car_width/2),
                car_length, car_width,
                facecolor=colors[i], alpha=0.5, edgecolor='black', linewidth=1.5, 
                linestyle='--', zorder=5
            )
            ax.add_patch(end_rect)
    
    # 起点编队框
    start_xs = [t[0][0] for t in traj if t]
    start_ys = [t[0][1] for t in traj if t]
    formation_rect = Rectangle(
        (min(start_xs) - 0.3, min(start_ys) - 0.3),
        max(start_xs) - min(start_xs) + 0.6,
        max(start_ys) - min(start_ys) + 0.6,
        fill=False, edgecolor='purple', linewidth=2, linestyle=':', zorder=4,
        label='Initial Formation'
    )
    ax.add_patch(formation_rect)
    
    # 目标点
    ax.scatter([goal[0]], [goal[1]], c='gold', marker='*', s=300, 
               label='Goal', zorder=10, edgecolors='black')
    
    ax.set_xlim(-2, max_x + 5)
    ax.set_ylim(-road_half_width - 1, road_half_width + 1)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Baseline2: Rule-based Formation Switching (DWA Control)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'formation_switch_demo.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_baseline2_demo(
    out_dir: str,
    max_steps: int = 800,
    dt: float = 0.1,
    road_half_width: float = 2.5,
    obstacle_mode: str = "gap",
):
    """原版baseline2演示（带障碍物触发变阵）"""
    os.makedirs(out_dir, exist_ok=True)

    vehicle_params = VehicleParams()
    hp = Baseline2HyperParams()

    # Scenario: empty road + optional gate to trigger reconfiguration
    obstacles: List[Obstacle] = []
    if obstacle_mode == "gap":
        # A "narrow gate" that is narrower than rectangle formation but wider than single ugv.
        upper = Obstacle(x=18.0, y=1.65, radius=1.0, obs_type="rect", width=6.0, height=1.7)
        lower = Obstacle(x=18.0, y=-1.65, radius=1.0, obs_type="rect", width=6.0, height=1.7)
        obstacles = [upper, lower]

    # Initial states: 4 vehicles in rectangle formation
    cars: List[VehicleState] = [
        VehicleState(x=0.0, y=0.6, theta=0.0, v=0.8),
        VehicleState(x=0.0, y=-0.6, theta=0.0, v=0.8),
        VehicleState(x=-1.5, y=0.6, theta=0.0, v=0.8),
        VehicleState(x=-1.5, y=-0.6, theta=0.0, v=0.8),
    ]

    bicycle = BicycleModel(vehicle_params)

    # DWA for MODE_AVOID
    dwa_params = DWAParams(
        predict_time=2.0,
        dt=dt,
        min_speed=0.3,
        max_speed=1.8,
        rl_direction_weight=6.0,
    )
    dwa = DWAController(vehicle_params, dwa_params)
    dwa.road_half_width = road_half_width

    goal = (35.0, 0.0)

    # Logs
    traj = [[] for _ in range(4)]
    mode_log: List[Mode] = []
    reason_log: List[str] = []

    last_mode: Optional[Mode] = None

    for step in range(max_steps):
        leader = cars[0]

        # Compute scene metrics
        D0 = min_distance_to_obstacles(leader, obstacles, vehicle_params)
        d_gap = estimate_gap_width(leader.x, obstacles, road_half_width, vehicle_params, hp)
        H_form_rect = formation_width(hp.rect_offsets, vehicle_params)
        h_ugv = float(vehicle_params.car_width)

        metrics = SceneMetrics(D0=D0, d_gap=d_gap, H_form=H_form_rect, h_ugv=h_ugv)
        mode, reason = choose_mode(metrics, hp)

        mode_log.append(mode)
        reason_log.append(reason)

        if mode != last_mode:
            print(f"[Step {step}] mode={mode} reason={reason} D0={D0:.2f} gap={d_gap:.2f} H_form={H_form_rect:.2f} h_ugv={h_ugv:.2f}")
            last_mode = mode

        # Select target formation offsets
        if mode == Mode.MODE_RECONFIGURE:
            offsets = hp.line_offsets
        else:
            offsets = hp.rect_offsets

        # Control
        new_cars: List[VehicleState] = []

        if mode == Mode.MODE_AVOID:
            for i, car in enumerate(cars):
                desired_y = cars[0].y + offsets[i][1]
                preferred_y = float(np.clip(desired_y, -road_half_width + 0.2, road_half_width - 0.2))
                control_state = (car.x, car.y, car.theta, car.v)
                a, delta = dwa.compute(control_state, goal, obstacles, preferred_y)
                new_cars.append(bicycle.step(car, (a, delta), dt))
        else:
            a0, d0 = leader_control_to_goal(cars[0], goal, hp)
            new_cars.append(bicycle.step(cars[0], (a0, d0), dt))

            for i in range(1, 4):
                ai, di = follower_virtual_structure_control(cars[i], cars[0], offsets[i], hp)
                new_cars.append(bicycle.step(cars[i], (ai, di), dt))

        cars = new_cars

        for i, s in enumerate(cars):
            traj[i].append((s.x, s.y))

        if np.mean([c.x for c in cars]) >= goal[0]:
            break

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))

    for obs in obstacles:
        if obs.width > 0 and obs.height > 0:
            rect = plt.Rectangle(
                (obs.x - obs.width / 2, obs.y - obs.height / 2),
                obs.width,
                obs.height,
                color="gray",
                alpha=0.6,
            )
            ax.add_patch(rect)
        else:
            circ = plt.Circle((obs.x, obs.y), obs.radius, color="gray", alpha=0.6)
            ax.add_patch(circ)

    ax.axhline(y=road_half_width, color="black", linestyle="--")
    ax.axhline(y=-road_half_width, color="black", linestyle="--")

    colors = ["red", "blue", "green", "orange"]
    for i in range(4):
        if traj[i]:
            xs, ys = zip(*traj[i])
            ax.plot(xs, ys, color=colors[i], label=f"car{i}")

    ax.scatter([goal[0]], [goal[1]], c="gold", marker="*", s=200, label="goal")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_title("Baseline2 Rule-based FSM Demo")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "baseline2_demo.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save mode log
    with open(os.path.join(out_dir, "mode_log.txt"), "w", encoding="utf-8") as f:
        for i, (m, r) in enumerate(zip(mode_log, reason_log)):
            f.write(f"{i}\t{m}\t{r}\n")


def run_double_narrow_demo(
    out_dir: str,
    max_steps: int = 600,
    dt: float = 0.1,
    road_half_width: float = 2.5,
):
    """
    双窄门场景测试 (double_narrow)
    - 第一门: x=25m, 通道宽1.6m
    - 第二门: x=55m, 通道宽1.0m (极窄)
    
    要求: 无碰撞通过
    """
    from matplotlib.patches import Rectangle
    
    os.makedirs(out_dir, exist_ok=True)

    vehicle_params = VehicleParams()
    hp = Baseline2HyperParams()
    car_radius = vehicle_params.car_radius

    # 从CONFIG导入double_narrow场景
    from config import OBSTACLE_SCENARIOS
    obstacles: List[Obstacle] = OBSTACLE_SCENARIOS["double_narrow"]
    
    # 初始位置 - 2x2矩形队形（缩小y偏移以适应窄门）
    car_states = [
        {'x': 5.0, 'y': 0.8, 'v': hp.v_ref},   # Car 0: 前左 (y正)
        {'x': 5.0, 'y': -0.8, 'v': hp.v_ref},  # Car 1: 前右 (y负)
        {'x': 3.0, 'y': 0.8, 'v': hp.v_ref},   # Car 2: 后左 (y正)
        {'x': 3.0, 'y': -0.8, 'v': hp.v_ref},  # Car 3: 后右 (y负)
    ]

    goal_x = 75.0
    
    # 窄门位置: x=25(1.6m宽), x=55(1.0m宽)
    # 扩大安全区间，确保完全离开后再恢复
    narrow_zones = [(22, 32), (50, 65)]  # 扩大安全范围
    lookahead = 6.0  # 缩短提前变阵距离

    # Logs
    traj = [[] for _ in range(4)]
    collision_log = []  # 碰撞记录
    
    # 状态机
    current_mode = "RECT"
    
    # 速度参数
    v_fast = hp.v_ref * 1.8
    v_slow = hp.v_ref * 0.2  # 更慢以避免碰撞
    v_normal = hp.v_ref

    print("="*60)
    print("双窄门场景测试 (double_narrow)")
    print(f"第一门: x=25m, 通道宽1.6m (障碍物占y=[0.8,2.5]和y=[-2.5,-0.8])")
    print(f"第二门: x=55m, 通道宽1.0m (障碍物占y=[0.5,2.5]和y=[-2.5,-0.5])")
    print(f"车辆半径: {car_radius:.3f}m")
    print(f"变阵方法: 速度差 (y正加速, y负减速)")
    print("="*60)
    
    # 详细日志
    min_dist_to_obs_log = []  # 记录每步最小距离
    
    def check_collision_detailed(car_states, obstacles, car_radius):
        """检测碰撞：车车碰撞+车障碍物碰撞，返回详细信息"""
        collisions = []
        min_dist_to_obs = float('inf')
        min_dist_between_cars = float('inf')
        
        # 车车碰撞
        for i in range(4):
            for j in range(i+1, 4):
                dx = car_states[i]['x'] - car_states[j]['x']
                dy = car_states[i]['y'] - car_states[j]['y']
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist_between_cars = min(min_dist_between_cars, dist)
                if dist < car_radius * 2:
                    collisions.append(f"Car{i}-Car{j}(d={dist:.3f})")
        
        # 车障碍物碰撞
        for i, car in enumerate(car_states):
            for obs in obstacles:
                if obs.obs_type == "rect":
                    ox_min = obs.x - obs.width/2
                    ox_max = obs.x + obs.width/2
                    oy_min = obs.y - obs.height/2
                    oy_max = obs.y + obs.height/2
                    closest_x = max(ox_min, min(car['x'], ox_max))
                    closest_y = max(oy_min, min(car['y'], oy_max))
                    dx = car['x'] - closest_x
                    dy = car['y'] - closest_y
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist_to_obs = min(min_dist_to_obs, dist)
                    if dist < car_radius:
                        collisions.append(f"Car{i}-Obs(d={dist:.3f})")
        
        return collisions, min_dist_to_obs, min_dist_between_cars

    for step in range(max_steps):
        avg_x = sum(c['x'] for c in car_states) / 4
        
        # 检测是否需要变阵
        in_narrow_zone = False
        approaching_narrow = False
        
        for zone_start, zone_end in narrow_zones:
            if zone_start <= avg_x <= zone_end:
                in_narrow_zone = True
                break
            elif zone_start - lookahead <= avg_x < zone_start:
                approaching_narrow = True
                break
        
        # 状态转换
        if current_mode == "RECT" and approaching_narrow:
            current_mode = "TO_LINE"
            print(f"[Step {step}] x={avg_x:.1f}m: 开始变阵 -> LINE")
        elif current_mode == "TO_LINE":
            y_pos_min_x = min(car_states[0]['x'], car_states[2]['x'])
            y_neg_max_x = max(car_states[1]['x'], car_states[3]['x'])
            if y_pos_min_x > y_neg_max_x + 1.5:
                current_mode = "LINE"
                print(f"[Step {step}] x={avg_x:.1f}m: 变阵完成 -> LINE")
        elif current_mode == "LINE" and not in_narrow_zone and not approaching_narrow:
            current_mode = "TO_RECT"
            print(f"[Step {step}] x={avg_x:.1f}m: 开始恢复 -> RECT")
        elif current_mode == "TO_RECT":
            if abs(car_states[0]['x'] - car_states[1]['x']) < 0.5 and \
               abs(car_states[2]['x'] - car_states[3]['x']) < 0.5:
                current_mode = "RECT"
                print(f"[Step {step}] x={avg_x:.1f}m: 恢复完成 -> RECT")
        
        # 控制
        for i, car in enumerate(car_states):
            is_y_positive = (i == 0 or i == 2)
            
            if current_mode == "RECT":
                car['v'] = v_normal
                target_y = 0.6 if is_y_positive else -0.6  # 缩小y偏移以适应窄门
            elif current_mode == "TO_LINE":
                car['v'] = v_fast if is_y_positive else v_slow
                target_y = 0.15 if is_y_positive else -0.15  # 更窄的线
            elif current_mode == "LINE":
                car['v'] = v_normal
                target_y = 0.0
            elif current_mode == "TO_RECT":
                car['v'] = v_slow if is_y_positive else v_fast
                target_y = 0.6 if is_y_positive else -0.6  # 缩小y偏移
            
            car['x'] += car['v'] * dt
            car['y'] += (target_y - car['y']) * 0.1

        # 检测碰撞
        collisions, min_d_obs, min_d_cars = check_collision_detailed(car_states, obstacles, car_radius)
        min_dist_to_obs_log.append((avg_x, min_d_obs, min_d_cars))
        if collisions:
            collision_log.append((step, avg_x, collisions))

        for i, car in enumerate(car_states):
            traj[i].append((car['x'], car['y']))

        if avg_x >= goal_x:
            break
    
    # 输出详细结果
    print("\n" + "="*60)
    print("详细验证结果:")
    print("="*60)
    
    # 找到在障碍物区域内的最小距离
    critical_zones = [(22, 28), (50, 60)]
    for zone_id, (z_start, z_end) in enumerate(critical_zones):
        zone_data = [(x, d_obs, d_cars) for x, d_obs, d_cars in min_dist_to_obs_log if z_start <= x <= z_end]
        if zone_data:
            min_d_obs_in_zone = min(d[1] for d in zone_data)
            min_d_cars_in_zone = min(d[2] for d in zone_data)
            print(f"\n窄门{zone_id+1} (x={z_start}-{z_end}m):")
            print(f"  最小车-障碍物距离: {min_d_obs_in_zone:.4f}m (安全阈值: {car_radius:.4f}m)")
            print(f"  最小车-车距离: {min_d_cars_in_zone:.4f}m (安全阈值: {car_radius*2:.4f}m)")
            if min_d_obs_in_zone > car_radius:
                print(f"  [OK] car-obs SAFE (margin: {min_d_obs_in_zone - car_radius:.4f}m)")
            else:
                print(f"  [FAIL] car-obs COLLISION!")
            if min_d_cars_in_zone > car_radius*2:
                print(f"  [OK] car-car SAFE (margin: {min_d_cars_in_zone - car_radius*2:.4f}m)")
            else:
                print(f"  [FAIL] car-car COLLISION!")
    
    if collision_log:
        print(f"\n[警告] 发生 {len(collision_log)} 次碰撞:")
        for step, x, cols in collision_log[:10]:
            print(f"  Step {step}, x={x:.1f}m: {cols}")
    else:
        print(f"\n[成功] 全程无碰撞!")

    # 绘图
    fig, ax = plt.subplots(figsize=(20, 5))
    
    car_length = vehicle_params.car_length
    car_width = vehicle_params.car_width
    
    # 道路
    max_x = max(max(p[0] for p in t) for t in traj if t) + 5
    ax.fill_between([0, max_x], [-road_half_width]*2, [road_half_width]*2, 
                    color='lightgray', alpha=0.3)
    ax.axhline(y=road_half_width, color='black', linestyle='--', linewidth=2)
    ax.axhline(y=-road_half_width, color='black', linestyle='--', linewidth=2)
    
    # 障碍物
    for obs in obstacles:
        rect = Rectangle(
            (obs.x - obs.width/2, obs.y - obs.height/2),
            obs.width, obs.height,
            facecolor='red', alpha=0.7, edgecolor='black'
        )
        ax.add_patch(rect)
    
    # 轨迹
    colors = ['red', 'blue', 'green', 'orange']
    linestyles = ['-', '--', '-.', ':']
    
    for i, t in enumerate(traj):
        if t:
            xs, ys = zip(*t)
            ax.plot(xs, ys, color=colors[i], linestyle=linestyles[i], 
                    label=f'Car {i}', linewidth=1.5, alpha=0.85)
            # 起点
            start_rect = Rectangle(
                (xs[0] - car_length/2, ys[0] - car_width/2),
                car_length, car_width, 
                facecolor=colors[i], alpha=0.9, edgecolor='black', linewidth=1.5, zorder=5
            )
            ax.add_patch(start_rect)
            # 终点
            end_rect = Rectangle(
                (xs[-1] - car_length/2, ys[-1] - car_width/2),
                car_length, car_width,
                facecolor=colors[i], alpha=0.5, edgecolor='black', linewidth=1.5, 
                linestyle='--', zorder=5
            )
            ax.add_patch(end_rect)
    
    ax.set_xlim(-2, max_x + 5)
    ax.set_ylim(-road_half_width - 1, road_half_width + 1)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    collision_status = '✅ No Collision' if not collision_log else f'❌ {len(collision_log)} Collisions'
    ax.set_title(f'Baseline2: Double Narrow ({collision_status})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'double_narrow_demo.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n双窄门演示完成！输出保存到: {out_dir}")
    return len(collision_log) == 0  # 返回是否无碰撞


def run_single_narrow_test(
    out_dir: str,
    gate_type: str = "wide",  # "wide" (1.6m) or "narrow" (1.0m)
    max_steps: int = 400,
    dt: float = 0.1,
    road_half_width: float = 2.5,
):
    """
    单窄门测试
    - wide: 1.6m通道 (x=25m)
    - narrow: 1.0m通道 (x=25m)
    """
    from matplotlib.patches import Rectangle
    
    os.makedirs(out_dir, exist_ok=True)
    vehicle_params = VehicleParams()
    hp = Baseline2HyperParams()
    car_radius = vehicle_params.car_radius

    # 根据类型选择障碍物 (长度6m)
    if gate_type == "wide":
        obstacles = [
            Obstacle(20, 1.7, 0.9, "rect", 6.0, 1.8),
            Obstacle(20, -1.7, 0.9, "rect", 6.0, 1.8),
        ]
        gate_width = 1.6
        gate_name = "Wide Gate"
    else:
        obstacles = [
            Obstacle(20, 1.5, 1.0, "rect", 6.0, 2.0),
            Obstacle(20, -1.5, 1.0, "rect", 6.0, 2.0),
        ]
        gate_width = 1.0
        gate_name = "Narrow Gate"
    
    # 根据通道宽度调整初始位置
    if gate_type == "narrow":
        # 1.0m通道，初始y要更小
        init_y = 0.4
        lookahead = 8.0  # 更早变阵
    else:
        init_y = 0.6
        lookahead = 5.0
    
    car_states = [
        {'x': 5.0, 'y': init_y, 'v': hp.v_ref},
        {'x': 5.0, 'y': -init_y, 'v': hp.v_ref},
        {'x': 3.0, 'y': init_y, 'v': hp.v_ref},
        {'x': 3.0, 'y': -init_y, 'v': hp.v_ref},
    ]

    goal_x = 35.0
    # 障碍物从 x=17到x=23 (6m长)
    narrow_zones = [(17, 26)] if gate_type == "narrow" else [(17, 25)]
    
    traj = [[] for _ in range(4)]
    collision_log = []
    min_dist_log = []
    current_mode = "RECT"
    
    v_fast = hp.v_ref * 1.8
    v_slow = hp.v_ref * 0.2
    v_normal = hp.v_ref

    print(f"\n{'='*60}")
    print(f"单窄门测试: {gate_name}")
    print(f"车辆半径: {car_radius:.3f}m")
    print(f"{'='*60}")
    
    def check_collision_detailed(car_states, obstacles, car_radius):
        collisions = []
        min_dist_to_obs = float('inf')
        min_dist_between_cars = float('inf')
        for i in range(4):
            for j in range(i+1, 4):
                dx = car_states[i]['x'] - car_states[j]['x']
                dy = car_states[i]['y'] - car_states[j]['y']
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist_between_cars = min(min_dist_between_cars, dist)
                if dist < car_radius * 2:
                    collisions.append(f"Car{i}-Car{j}")
        for i, car in enumerate(car_states):
            for obs in obstacles:
                if obs.obs_type == "rect":
                    ox_min = obs.x - obs.width/2
                    ox_max = obs.x + obs.width/2
                    oy_min = obs.y - obs.height/2
                    oy_max = obs.y + obs.height/2
                    closest_x = max(ox_min, min(car['x'], ox_max))
                    closest_y = max(oy_min, min(car['y'], oy_max))
                    dx = car['x'] - closest_x
                    dy = car['y'] - closest_y
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist_to_obs = min(min_dist_to_obs, dist)
                    if dist < car_radius:
                        collisions.append(f"Car{i}-Obs")
        return collisions, min_dist_to_obs, min_dist_between_cars

    for step in range(max_steps):
        avg_x = sum(c['x'] for c in car_states) / 4
        
        in_narrow = any(z[0] <= avg_x <= z[1] for z in narrow_zones)
        approaching = any(z[0] - lookahead <= avg_x < z[0] for z in narrow_zones)
        
        if current_mode == "RECT" and approaching:
            current_mode = "TO_LINE"
            print(f"[Step {step}] x={avg_x:.1f}m: RECT -> TO_LINE")
        elif current_mode == "TO_LINE":
            y_pos_min_x = min(car_states[0]['x'], car_states[2]['x'])
            y_neg_max_x = max(car_states[1]['x'], car_states[3]['x'])
            if y_pos_min_x > y_neg_max_x + 1.5:
                current_mode = "LINE"
                print(f"[Step {step}] x={avg_x:.1f}m: TO_LINE -> LINE")
        elif current_mode == "LINE" and not in_narrow and not approaching:
            current_mode = "TO_RECT"
            print(f"[Step {step}] x={avg_x:.1f}m: LINE -> TO_RECT")
        elif current_mode == "TO_RECT":
            if abs(car_states[0]['x'] - car_states[1]['x']) < 0.5:
                current_mode = "RECT"
                print(f"[Step {step}] x={avg_x:.1f}m: TO_RECT -> RECT")
        
        for i, car in enumerate(car_states):
            is_y_pos = (i == 0 or i == 2)
            if current_mode == "RECT":
                car['v'] = v_normal
                target_y = 0.6 if is_y_pos else -0.6
            elif current_mode == "TO_LINE":
                car['v'] = v_fast if is_y_pos else v_slow
                target_y = 0.15 if is_y_pos else -0.15
            elif current_mode == "LINE":
                car['v'] = v_normal
                target_y = 0.0
            elif current_mode == "TO_RECT":
                car['v'] = v_slow if is_y_pos else v_fast
                target_y = 0.6 if is_y_pos else -0.6
            car['x'] += car['v'] * dt
            car['y'] += (target_y - car['y']) * 0.1

        collisions, min_d_obs, min_d_cars = check_collision_detailed(car_states, obstacles, car_radius)
        min_dist_log.append((avg_x, min_d_obs, min_d_cars))
        if collisions:
            collision_log.append((step, avg_x, collisions))
        for i, car in enumerate(car_states):
            traj[i].append((car['x'], car['y']))
        if avg_x >= goal_x:
            break
    
    # 验证结果
    zone_data = [(x, d_obs, d_cars) for x, d_obs, d_cars in min_dist_log if 22 <= x <= 30]
    if zone_data:
        min_d_obs = min(d[1] for d in zone_data)
        min_d_cars = min(d[2] for d in zone_data)
        print(f"\n窄门区域 (x=22-30m):")
        print(f"  最小车-障碍物距离: {min_d_obs:.4f}m (safety: {car_radius:.4f}m)")
        print(f"  [{'OK' if min_d_obs > car_radius else 'FAIL'}] margin: {min_d_obs - car_radius:.4f}m")
    
    success = len(collision_log) == 0
    if collision_log:
        print(f"\n碰撞详情 ({len(collision_log)}次):")
        for step, x, cols in collision_log[:5]:
            print(f"  Step {step}, x={x:.1f}m: {cols}")
    print(f"\n结果: {'PASS - 无碰撞' if success else 'FAIL - 有碰撞'}")
    
    # 绘图 - 缩短图片尺寸适合论文
    fig, ax = plt.subplots(figsize=(8, 2.5))
    car_length = vehicle_params.car_length
    car_width = vehicle_params.car_width
    max_x = max(max(p[0] for p in t) for t in traj if t) + 3
    
    ax.fill_between([0, max_x], [-road_half_width]*2, [road_half_width]*2, color='lightgray', alpha=0.3)
    ax.axhline(y=road_half_width, color='black', linestyle='--', linewidth=2)
    ax.axhline(y=-road_half_width, color='black', linestyle='--', linewidth=2)
    
    for obs in obstacles:
        rect = Rectangle((obs.x - obs.width/2, obs.y - obs.height/2), obs.width, obs.height,
                          facecolor='gray', alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, t in enumerate(traj):
        if t:
            xs, ys = zip(*t)
            ax.plot(xs, ys, color=colors[i], label=f'Car {i}', linewidth=1.5)
            ax.add_patch(Rectangle((xs[0]-car_length/2, ys[0]-car_width/2), car_length, car_width,
                                    facecolor=colors[i], alpha=0.9, edgecolor='black', zorder=5))
            ax.add_patch(Rectangle((xs[-1]-car_length/2, ys[-1]-car_width/2), car_length, car_width,
                                    facecolor=colors[i], alpha=0.5, edgecolor='black', linestyle='--', zorder=5))
    
    ax.set_xlim(0, max_x + 1)
    ax.set_ylim(-road_half_width - 0.5, road_half_width + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{gate_name}', fontweight='bold', fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # 输出PDF和PNG
    plt.savefig(os.path.join(out_dir, f'single_gate_{gate_type}.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, f'single_gate_{gate_type}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return success


if __name__ == "__main__":
    out = os.path.join(CURRENT_DIR, "outputs")
    
    # 测试1: 宽窄门 (1.6m)
    print("\n" + "="*60)
    print("测试1: 宽窄门 (1.6m)")
    print("="*60)
    s1 = run_single_narrow_test(out_dir=out, gate_type="wide")
    
    # 测试2: 窄窄门 (1.0m)
    print("\n" + "="*60)
    print("测试2: 窄窄门 (1.0m)")
    print("="*60)
    s2 = run_single_narrow_test(out_dir=out, gate_type="narrow")
    
    print("\n" + "="*60)
    print("测试结果汇总:")
    print(f"  宽窄门(1.6m): {'PASS' if s1 else 'FAIL'}")
    print(f"  窄窄门(1.0m): {'PASS' if s2 else 'FAIL'}")
    print("="*60)
    
    print(f"\nDone. All outputs saved to: {out}")
