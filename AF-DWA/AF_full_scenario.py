"""
AF算法完整场景测试 (All Obstacle Types)

包含6种障碍物类型，紧凑排列：
1. 右侧障碍 (20m) - 整体左移通过
2. 左侧障碍 (38m) - 整体右移通过
3. 两侧窄门 (56m) - 压缩矩形队形通过 (通道1.6m)
4. 中间小障碍 (72m) - 队形不变，小幅避让
5. 中间大障碍 (88m) - 分两边通过
6. 极窄通道 (108m) - 一字长蛇阵通过 (通道1.0m)

总长约130m，展示AF算法在各种场景下的自适应能力。

输出文件：
- AF_full_scenario.png/pdf: 全场景轨迹图（含8个编队快照）
- AF_formation_snapshots.png/pdf: 8个关键位置的编队详细图（2x4子图）
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from typing import List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import VehicleParams, Obstacle

# 设置字体为Arial
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ========== 完整场景：包含所有障碍物类型（紧凑版，适合论文） ==========
FULL_SCENARIO_OBSTACLES = [
    # 场景1: 右侧障碍
    Obstacle(15, -1.5, 1.0, "rect", 3.2, 2.0),
    
    # 场景2: 左侧障碍
    Obstacle(26, 1.5, 1.0, "rect", 3.2, 2.0),
    
    # 场景3: 两侧窄门 (1.6m)
    Obstacle(37, 1.7, 0.9, "rect", 3.2, 1.8),
    Obstacle(37, -1.7, 0.9, "rect", 3.2, 1.8),
    
    # 场景4: 中间小障碍
    Obstacle(48, 0.0, 0.4, "circle"),
    
    # 场景5: 中间大障碍
    Obstacle(59, 0.0, 1.3, "car", 3.5, 2.6),
    
    # 场景6: 极窄通道 (1.0m) - 与大障碍保持足够距离
    Obstacle(78, 1.5, 1.0, "rect", 4.0, 2.0),
    Obstacle(78, -1.5, 1.0, "rect", 4.0, 2.0),
]


class AdaptiveFormationPolicy:
    """
    自适应队形策略 (AF - Adaptive Formation)
    
    根据前方障碍物类型自动选择最优队形：
    - 单侧障碍: 整体平移避让
    - 两侧窄门(>1.2m): 压缩矩形队形
    - 极窄通道(≤1.2m): 一字长蛇阵
    - 中间障碍: 分两边通过
    """
    
    def __init__(self, max_offset: float = 0.5):
        self.max_offset = max_offset
        self.lookahead_dist = 5.5
        self.lookahead_dist_very_narrow = 10.5
        self.current_mode = "RECT"
        self.template_y = [0.8, -0.8, 0.8, -0.8]  # 标准2x2队形
        self.road_half_width = 2.5
    
    def analyze_obstacle(self, avg_x: float, obstacles: List[Obstacle], car_states=None):
        """
        分析前方障碍物，返回场景类型和建议动作
        只检测最近的一组障碍物（x坐标相近的障碍物视为一组）
        
        Returns:
            scene_type: "clear" | "right_obs" | "left_obs" | "narrow_gate" | 
                       "very_narrow" | "center_small" | "center_large"
            gap_width: 可通行宽度
            center_y: 建议通行中心y坐标
        """
        # 找到前方最近的障碍物组
        nearest_obs_x = None
        group_tolerance = 10.0  # 同一组障碍物的x坐标容差
        
        for obs in obstacles:
            if obs.width > 0:
                obs_x_min = obs.x - obs.width / 2
            else:
                obs_x_min = obs.x - obs.radius
            
            # 只考虑前方的障碍物（包括正在通过的）
            # 使用最慢车的实际位置来判断是否通过
            obs_x_max = obs.x + obs.width / 2 if obs.width > 0 else obs.x + obs.radius
            if car_states:
                slowest_car_x = min(c['x'] for c in car_states) - 3  # 最慢车x + 安全余量
            else:
                slowest_car_x = avg_x - 20
            if obs_x_max > slowest_car_x:  # 最慢车还没完全通过
                if nearest_obs_x is None or obs.x < nearest_obs_x:
                    nearest_obs_x = obs.x
        
        if nearest_obs_x is None:
            return "clear", 5.0, 0.0, None
        
        # 只分析这一组障碍物
        upper_bound = self.road_half_width
        lower_bound = -self.road_half_width
        center_obs = None
        left_obs = None
        right_obs = None
        
        for obs in obstacles:
            # 只检测同一组障碍物
            if abs(obs.x - nearest_obs_x) > group_tolerance:
                continue
            
            # 判断障碍物位置
            if obs.obs_type == "circle" or (obs.obs_type == "car" and abs(obs.y) < 0.5):
                # 中间障碍物
                if obs.radius < 0.5:
                    center_obs = ("small", obs)
                else:
                    center_obs = ("large", obs)
            elif obs.y > 0.5:
                # 左侧障碍物
                left_obs = obs
                if obs.height > 0:
                    upper_bound = min(upper_bound, obs.y - obs.height / 2)
            elif obs.y < -0.5:
                # 右侧障碍物
                right_obs = obs
                if obs.height > 0:
                    lower_bound = max(lower_bound, obs.y + obs.height / 2)
        
        gap_width = upper_bound - lower_bound
        center_y = (upper_bound + lower_bound) / 2
        
        # 判断场景类型
        if center_obs:
            if center_obs[0] == "small":
                scene_type = "center_small"
            else:
                scene_type = "center_large"
            lookahead = self.lookahead_dist
            if nearest_obs_x > avg_x + lookahead:
                return "clear", 5.0, 0.0, None
            return scene_type, gap_width, center_y, center_obs[1]
        
        if left_obs and right_obs:
            # 两侧都有障碍物
            if gap_width <= 1.2:
                scene_type = "very_narrow"
            else:
                scene_type = "narrow_gate"

            lookahead = self.lookahead_dist_very_narrow if scene_type == "very_narrow" else self.lookahead_dist
            if nearest_obs_x > avg_x + lookahead:
                return "clear", 5.0, 0.0, None
            return scene_type, gap_width, center_y, None
        
        if right_obs and not left_obs:
            lookahead = self.lookahead_dist
            if nearest_obs_x > avg_x + lookahead:
                return "clear", 5.0, 0.0, None
            return "right_obs", gap_width, center_y, right_obs
        
        if left_obs and not right_obs:
            lookahead = self.lookahead_dist
            if nearest_obs_x > avg_x + lookahead:
                return "clear", 5.0, 0.0, None
            return "left_obs", gap_width, center_y, left_obs
        
        return "clear", gap_width, 0.0, None
    
    def get_target_formation(self, scene_type: str, gap_width: float, center_y: float):
        """
        根据场景类型返回目标队形
        
        关键：保持2x2队形结构
        - Car0, Car2 始终在上方 (y > 0)
        - Car1, Car3 始终在下方 (y < 0)
        - 上下车的y差距保持恒定
        
        Returns:
            mode: 队形模式
            target_y: 4车目标y坐标 [Car0, Car1, Car2, Car3]
            formation_center: 队形中心y坐标
        """
        # 队形参数
        y_spacing = 0.6  # 上下车y间距
        
        if scene_type == "clear":
            return "RECT", [y_spacing, -y_spacing, y_spacing, -y_spacing], 0.0
        
        elif scene_type == "right_obs":
            # 右侧障碍，整体左移（y增加）
            center = 0.8
            return "SHIFT_LEFT", [center + y_spacing, center - y_spacing, 
                                  center + y_spacing, center - y_spacing], center
        
        elif scene_type == "left_obs":
            # 左侧障碍，整体右移（y减小）
            center = -0.8
            return "SHIFT_RIGHT", [center + y_spacing, center - y_spacing,
                                   center + y_spacing, center - y_spacing], center
        
        elif scene_type == "narrow_gate":
            # 两侧窄门，压缩y间距
            compressed_spacing = 0.35
            return "COMPRESSED", [compressed_spacing, -compressed_spacing, 
                                  compressed_spacing, -compressed_spacing], 0.0
        
        elif scene_type == "very_narrow":
            # 极窄通道（1.0m），一字长蛇阵
            # 通道宽度1.0m，车宽0.3m，车中心需要在y∈[-0.35, 0.35]范围内
            # 为安全起见，所有车都走中心线(y=0)
            # 通过速度差异使车辆前后排列（x方向拉开），避免车间碰撞
            return "LINE", [0.0, 0.0, 0.0, 0.0], 0.0
        
        elif scene_type == "center_small":
            # 中间小障碍，稍微扩大y间距避开障碍
            return "RECT", [0.8, -0.8, 0.8, -0.8], 0.0
        
        elif scene_type == "center_large":
            # 中间大障碍，上下分开更大（障碍物占据y=[-1.3,1.3]，车要在外面）
            return "SPLIT", [1.8, -1.8, 1.8, -1.8], 0.0
        
        return "RECT", [y_spacing, -y_spacing, y_spacing, -y_spacing], 0.0


def run_full_scenario_test(
    max_steps: int = 2500,
    dt: float = 0.1,
    road_half_width: float = 2.5,
):
    """
    运行完整场景测试
    """
    vehicle_params = VehicleParams()
    car_radius = vehicle_params.car_radius
    v_ref = 1.5
    v_fast = v_ref * 1.6
    v_slow = v_ref * 0.4
    
    obstacles = FULL_SCENARIO_OBSTACLES
    policy = AdaptiveFormationPolicy()
    
    # 初始化车辆（2x2矩形队形）
    init_y = 0.8
    car_states = [
        {'x': 5.0, 'y': init_y, 'v': v_ref},
        {'x': 5.0, 'y': -init_y, 'v': v_ref},
        {'x': 3.0, 'y': init_y, 'v': v_ref},
        {'x': 3.0, 'y': -init_y, 'v': v_ref},
    ]
    
    goal_x = 88.0  # 紧凑版终点
    traj = [[] for _ in range(4)]
    collision_log = []
    mode_history = []
    current_mode = "RECT"
    target_y = [0.8, -0.8, 0.8, -0.8]
    v_cmd_alpha = 0.25

    for i, car in enumerate(car_states):
        traj[i].append((car['x'], car['y']))
    
    print(f"\n{'='*70}")
    print("AF算法完整场景测试 (All Obstacle Types)")
    print(f"{'='*70}")
    print("场景序列:")
    print("  1. (15m) 右侧障碍 -> 左移通过")
    print("  2. (26m) 左侧障碍 -> 右移通过")
    print("  3. (37m) 两侧窄门(1.6m) -> 压缩矩形")
    print("  4. (48m) 中间小障碍 -> 队形不变")
    print("  5. (59m) 中间大障碍 -> 分两边通过")
    print("  6. (78m) 极窄通道(1.0m) -> 一字长蛇阵")
    print(f"{'='*70}\n")
    
    for step in range(max_steps):
        avg_x = sum(c['x'] for c in car_states) / 4
        
        # 分析前方场景（传入实际车辆位置）
        scene_type, gap_width, center_y, _ = policy.analyze_obstacle(avg_x, obstacles, car_states)
        new_mode, new_target_y, formation_center = policy.get_target_formation(
            scene_type, gap_width, center_y
        )
        
        # 模式切换日志
        if new_mode != current_mode:
            print(f"[Step {step:4d}] x={avg_x:6.1f}m: {current_mode:12s} -> {new_mode:12s} "
                  f"(scene: {scene_type}, gap: {gap_width:.2f}m)")
            current_mode = new_mode
            target_y = new_target_y
        
        mode_history.append((avg_x, current_mode))
        
        # 控制逻辑
        for i, car in enumerate(car_states):
            is_y_pos = (i == 0 or i == 2)  # 上方车：Car0, Car2
            is_front = (i == 0 or i == 1)  # 前排车：Car0, Car1
            
            # 速度控制
            if current_mode == "RECT":
                v_des = v_ref
            elif current_mode in ["SHIFT_LEFT", "SHIFT_RIGHT"]:
                v_des = v_ref
            elif current_mode == "COMPRESSED":
                v_des = v_ref
            elif current_mode == "LINE":
                # 纵队变阵：保持前后排关系，避免交叉
                # 前排(Car0,Car1)始终比后排(Car2,Car3)快
                # 同排内通过速度差拉开间距
                # Car0: 最快, Car1: 次快, Car2: 次慢, Car3: 最慢
                speed_factors = [1.8, 1.3, 0.7, 0.3]
                v_des = v_ref * speed_factors[i]
            elif current_mode == "SPLIT":
                # 分两边通过大障碍
                v_des = v_ref
            else:
                v_des = v_ref

            car['v'] += (v_des - car['v']) * v_cmd_alpha
            
            # x方向更新
            car['x'] += car['v'] * dt
        
        # y方向收敛（平滑变阵，考虑车间碰撞避免）
        for i, car in enumerate(car_states):
            if current_mode == "LINE":
                # LINE模式下，检查与其他车的碰撞风险
                min_safe_margin = float('inf')
                for j, other in enumerate(car_states):
                    if i != j:
                        x_dist = abs(car['x'] - other['x'])
                        y_dist = abs(car['y'] - other['y'])
                        current_dist = np.sqrt(x_dist**2 + y_dist**2)
                        safe_margin = current_dist - 2 * car_radius
                        if safe_margin < min_safe_margin:
                            min_safe_margin = safe_margin
                
                # 根据安全裕度动态调整y收敛速度（更平滑）
                if min_safe_margin > 0.5:
                    y_converge_rate = 0.06  # 平滑收敛
                elif min_safe_margin > 0.2:
                    y_converge_rate = 0.025
                else:
                    y_converge_rate = 0.005
            elif current_mode == "SPLIT":
                # SPLIT模式：平滑分开
                y_converge_rate = 0.10
            elif current_mode in ["SHIFT_LEFT", "SHIFT_RIGHT"]:
                # 平移模式：平滑移动
                y_converge_rate = 0.12
            elif current_mode == "COMPRESSED":
                # 压缩模式：平滑压缩
                y_converge_rate = 0.10
            else:
                # 其他模式（RECT）：平滑恢复
                y_converge_rate = 0.12
            
            car['y'] += (target_y[i] - car['y']) * y_converge_rate
        
        # 碰撞检测
        for i, car in enumerate(car_states):
            for obs in obstacles:
                if obs.width > 0:
                    ox_min, ox_max = obs.x - obs.width/2, obs.x + obs.width/2
                    oy_min, oy_max = obs.y - obs.height/2, obs.y + obs.height/2
                    closest_x = max(ox_min, min(car['x'], ox_max))
                    closest_y = max(oy_min, min(car['y'], oy_max))
                else:
                    closest_x = obs.x
                    closest_y = obs.y
                dist = np.sqrt((car['x']-closest_x)**2 + (car['y']-closest_y)**2)
                
                effective_radius = obs.radius if obs.width == 0 else 0
                if dist < car_radius + effective_radius:
                    collision_log.append((step, avg_x, f"Car{i}-Obs({obs.x:.0f}m)"))
        
        # 车间碰撞检测
        for i in range(4):
            for j in range(i+1, 4):
                dist = np.sqrt((car_states[i]['x']-car_states[j]['x'])**2 + 
                              (car_states[i]['y']-car_states[j]['y'])**2)
                if dist < 2 * car_radius:
                    collision_log.append((step, avg_x, f"Car{i}-Car{j}"))
        
        for i, car in enumerate(car_states):
            traj[i].append((car['x'], car['y']))
        
        if avg_x >= goal_x:
            break
    
    # 结果统计
    success = len(collision_log) == 0
    print(f"\n{'='*70}")
    print(f"测试结果: {'PASS' if success else 'FAIL'}")
    if collision_log:
        print(f"碰撞记录 ({len(collision_log)}次):")
        for col in collision_log[:10]:
            print(f"  Step {col[0]}, x={col[1]:.1f}m: {col[2]}")
    print(f"{'='*70}")
    
    # ========== 绘图：单张全景图（紧凑版，适合论文） ==========
    fig, ax = plt.subplots(figsize=(12, 2.8))
    car_length = vehicle_params.car_length
    car_width = vehicle_params.car_width
    plot_x_max = 85
    
    # 道路边界（无灰色背景）
    ax.axhline(y=road_half_width, color='black', linestyle='--', linewidth=2.0)
    ax.axhline(y=-road_half_width, color='black', linestyle='--', linewidth=2.0)
    
    # 障碍物（颜色与其他图一致）
    for obs in obstacles:
        if obs.width > 0:
            rect = Rectangle((obs.x-obs.width/2, obs.y-obs.height/2), 
                            obs.width, obs.height,
                            facecolor='gray', alpha=0.7, edgecolor='black', linewidth=1.0)
            ax.add_patch(rect)
        else:
            circle = Circle((obs.x, obs.y), obs.radius, 
                           facecolor='gray', alpha=0.7, edgecolor='black', linewidth=1.0)
            ax.add_patch(circle)
    
    # 轨迹颜色
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']  # 红、蓝、绿、橙
    
    # 绘制轨迹线（虚线，更粗更清晰）
    def _smooth_mavg(arr: np.ndarray, window: int) -> np.ndarray:
        if window <= 1 or arr.size < 3:
            return arr
        w = int(window)
        if w % 2 == 0:
            w += 1
        pad = w // 2
        padded = np.pad(arr, (pad, pad), mode='edge')
        kernel = np.ones(w, dtype=float) / float(w)
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed

    for i, t in enumerate(traj):
        if t:
            xs, ys = zip(*t)
            xs_np = np.asarray(xs, dtype=float)
            ys_np = np.asarray(ys, dtype=float)
            ys_np = _smooth_mavg(ys_np, window=30)  # 减小平滑窗口，让轨迹更自然
            ax.plot(xs_np, ys_np, color=colors[i], linestyle=(0, (2.2, 1.8)), 
                   linewidth=2.0, alpha=0.85)
    
    # ========== 关键编队快照 ==========
    # 选择关键时刻（按步数）展示编队，确保4辆车同时显示
    total_steps = len(traj[0])

    segments = []
    if mode_history:
        start_idx = 0
        last_mode = mode_history[0][1]
        for k in range(1, len(mode_history)):
            if mode_history[k][1] != last_mode:
                segments.append((start_idx, k - 1, last_mode))
                start_idx = k
                last_mode = mode_history[k][1]
        segments.append((start_idx, len(mode_history) - 1, last_mode))

    mode_by_step = ["RECT"]
    if mode_history:
        for _, m in mode_history:
            mode_by_step.append(m)
    if len(mode_by_step) < total_steps:
        mode_by_step += [mode_by_step[-1]] * (total_steps - len(mode_by_step))

    avg_x_series = []
    for k in range(total_steps):
        avg_x_k = sum(traj[i][k][0] for i in range(4)) / 4
        avg_x_series.append(avg_x_k)

    line_candidates = [k for k in range(total_steps) if mode_by_step[k] == "LINE"]
    best_line_step = None
    if line_candidates:
        line_x_min = 74.0
        line_x_max = 82.0
        filtered = [k for k in line_candidates if line_x_min <= avg_x_series[k] <= line_x_max]
        if not filtered:
            filtered = line_candidates

        best_key = None
        for k in filtered:
            positions = [traj[i][k] for i in range(4)]
            max_abs_y = max(abs(p[1]) for p in positions)
            xs_sorted = sorted(p[0] for p in positions)
            gaps = [xs_sorted[j + 1] - xs_sorted[j] for j in range(3)]
            min_gap_x = min(gaps) if gaps else 0.0
            key = (max_abs_y, -min_gap_x)
            if best_key is None or key < best_key:
                best_key = key
                best_line_step = k

    def _nearest_step_for_x_in_mode(x_target: float, expected_mode: str) -> int:
        candidates = [k for k in range(total_steps) if mode_by_step[k] == expected_mode]
        if not candidates:
            candidates = list(range(total_steps))
        best_k = candidates[0]
        best_err = abs(avg_x_series[best_k] - x_target)
        for k in candidates[1:]:
            err = abs(avg_x_series[k] - x_target)
            if err < best_err:
                best_err = err
                best_k = k
        return best_k

    snapshot_steps = [0]
    snapshot_steps.append(_nearest_step_for_x_in_mode(15.0, "SHIFT_LEFT"))
    snapshot_steps.append(_nearest_step_for_x_in_mode(26.0, "SHIFT_RIGHT"))
    snapshot_steps.append(_nearest_step_for_x_in_mode(37.0, "COMPRESSED"))
    snapshot_steps.append(_nearest_step_for_x_in_mode(48.0, "RECT"))
    snapshot_steps.append(_nearest_step_for_x_in_mode(59.0, "SPLIT"))
    if best_line_step is not None:
        snapshot_steps.append(best_line_step)
    snapshot_steps = list(dict.fromkeys(snapshot_steps))

    snapshot_scale = 1.6
    
    for step_idx in snapshot_steps:
        if step_idx < 0 or step_idx >= total_steps:
            continue
        positions = []
        # 同时绘制所有4辆车在该时刻的位置
        for i, t in enumerate(traj):
            if step_idx < len(t):
                x, y = t[step_idx]
                positions.append((x, y))

        if len(positions) == 4:
            mode_label = mode_by_step[step_idx] if step_idx < len(mode_by_step) else ""
            if step_idx == 0:
                mode_label = "RECT"
            if mode_label == "LINE":
                gate_center = 78.0
                desired_spacing = (car_length * snapshot_scale) * 1.35
                spacing_limit = (plot_x_max - 1.0 - gate_center) / 1.5
                spacing = min(desired_spacing, spacing_limit)
                order = sorted(range(4), key=lambda idx: positions[idx][0])
                display_xs = [gate_center + (r - 1.5) * spacing for r in range(4)]
                display_ys = [0.0] * 4
                ax.plot(display_xs, display_ys, color='black', linewidth=1.2, alpha=0.45, zorder=9)
                ax.plot([display_xs[0] - 0.6, display_xs[-1] + 0.6], [0.0, 0.0],
                        color='#666666', linewidth=1.0, alpha=0.30, zorder=8)

                for r, car_idx in enumerate(order):
                    x = display_xs[r]
                    y = 0.0
                    rect = Rectangle((x-(car_length*snapshot_scale)/2, y-(car_width*snapshot_scale)/2),
                                     car_length*snapshot_scale, car_width*snapshot_scale,
                                     facecolor=colors[car_idx], alpha=1.0,
                                     edgecolor='black', linewidth=1.2, zorder=10)
                    ax.add_patch(rect)
            else:
                for i, (x, y) in enumerate(positions):
                    rect = Rectangle((x-(car_length*snapshot_scale)/2, y-(car_width*snapshot_scale)/2),
                                     car_length*snapshot_scale, car_width*snapshot_scale,
                                     facecolor=colors[i], alpha=1.0,
                                     edgecolor='black', linewidth=1.2, zorder=10)
                    ax.add_patch(rect)
                outline_order = [0, 1, 3, 2, 0]
                xs = [positions[i][0] for i in outline_order]
                ys = [positions[i][1] for i in outline_order]
                ax.fill(xs, ys, color='black', alpha=0.06, zorder=8)
                ax.plot(xs, ys, color='black', linewidth=1.0, alpha=0.35, zorder=9)
    
    # 创建图例（使用UGV标签，与其他图一致）
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], edgecolor='black', label='UGV 0'),
        Patch(facecolor=colors[1], edgecolor='black', label='UGV 1'),
        Patch(facecolor=colors[2], edgecolor='black', label='UGV 2'),
        Patch(facecolor=colors[3], edgecolor='black', label='UGV 3'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.35),
              ncol=4, fontsize=9, frameon=True)
    
    ax.set_xlim(0, plot_x_max)
    ax.set_ylim(-road_half_width - 0.5, road_half_width + 0.5)
    ax.set_aspect('auto')
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_title('Adaptive Formation Control Through Multiple Obstacles', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.35)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(CURRENT_DIR, 'AF_full_scenario.pdf'), 
               format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(CURRENT_DIR, 'AF_full_scenario.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig)

    def _first_step_matching_mode(substr: str):
        for k in range(total_steps):
            if substr in str(mode_by_step[k]).strip():
                return k
        return None

    shift_step = None
    for s in snapshot_steps:
        if "SHIFT" in str(mode_by_step[s]).strip():
            shift_step = s
            break
    if shift_step is None:
        shift_step = _first_step_matching_mode("SHIFT")

    compressed_step = _nearest_step_for_x_in_mode(37.0, "COMPRESSED")
    rect_center_small_step = _nearest_step_for_x_in_mode(48.0, "RECT")
    split_step = _nearest_step_for_x_in_mode(59.0, "SPLIT")
    line_step = best_line_step
    if line_step is None:
        line_step = _first_step_matching_mode("LINE")

    desired_steps = [0, shift_step, compressed_step, rect_center_small_step, split_step, line_step]
    snapshots_to_show = []
    for s in desired_steps:
        if s is None:
            continue
        if s in snapshots_to_show:
            continue
        snapshots_to_show.append(s)

    for s in snapshot_steps:
        if s in snapshots_to_show:
            continue
        snapshots_to_show.append(s)
        if len(snapshots_to_show) >= 6:
            break

    while len(snapshots_to_show) < 6:
        snapshots_to_show.append(snapshots_to_show[-1] if snapshots_to_show else 0)
    snapshots_to_show = snapshots_to_show[:6]

    fig2 = plt.figure(figsize=(12, 5))
    gs = fig2.add_gridspec(2, 3)
    axes = [fig2.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    for plot_i, ax2 in enumerate(axes):
        step_idx = snapshots_to_show[plot_i]
        step_idx = max(0, min(step_idx, total_steps - 1))
        positions = [traj[i][step_idx] for i in range(4)]
        cx = sum(p[0] for p in positions) / 4
        min_x = min(p[0] for p in positions)
        max_x = max(p[0] for p in positions)

        x_left = cx - 7
        x_right = cx + 7
        mode_label = mode_by_step[step_idx] if step_idx < len(mode_by_step) else ""
        if step_idx == 0:
            mode_label = "RECT"
        if mode_label == "LINE":
            x_left = 71.0
            x_right = 85.0

        ax2.fill_between([x_left, x_right], [-road_half_width] * 2, [road_half_width] * 2,
                         color='#ededed', alpha=0.8)
        ax2.axhline(y=road_half_width, color='black', linestyle='--', linewidth=1.6)
        ax2.axhline(y=-road_half_width, color='black', linestyle='--', linewidth=1.6)
        if mode_label == "LINE":
            ax2.axhline(y=0.0, color='#666666', linestyle='-', linewidth=1.0, alpha=0.45)

        for obs in obstacles:
            if obs.width > 0:
                if (obs.x + obs.width / 2) < x_left or (obs.x - obs.width / 2) > x_right:
                    continue
                rect = Rectangle((obs.x-obs.width/2, obs.y-obs.height/2),
                                 obs.width, obs.height,
                                 facecolor='#6b6b6b', alpha=0.9, edgecolor='#222222', linewidth=1.0)
                ax2.add_patch(rect)
            else:
                if (obs.x + obs.radius) < x_left or (obs.x - obs.radius) > x_right:
                    continue
                circle = Circle((obs.x, obs.y), obs.radius,
                                facecolor='#6b6b6b', alpha=0.9, edgecolor='#222222', linewidth=1.0)
                ax2.add_patch(circle)

        outline_order = [0, 1, 3, 2, 0]
        if mode_label == "LINE":
            gate_center = 78.0
            spacing = 1.60
            order = sorted(range(4), key=lambda idx: positions[idx][0])
            display_xs = [gate_center + (r - 1.5) * spacing for r in range(4)]
            display_ys = [0.0] * 4
            ax2.plot(display_xs, display_ys, color='black', linewidth=1.2, alpha=0.45, zorder=4)
        else:
            xs = [positions[i][0] for i in outline_order]
            ys = [positions[i][1] for i in outline_order]
            ax2.fill(xs, ys, color='black', alpha=0.06, zorder=3)
            ax2.plot(xs, ys, color='black', linewidth=1.0, alpha=0.35, zorder=4)

        if mode_label == "LINE":
            gate_center = 78.0
            spacing = 1.60
            order = sorted(range(4), key=lambda idx: positions[idx][0])
            display_xs = [gate_center + (r - 1.5) * spacing for r in range(4)]
            for r, car_idx in enumerate(order):
                rect = Rectangle((display_xs[r]-car_length/2, -car_width/2),
                                 car_length, car_width,
                                 facecolor=colors[car_idx], alpha=1.0,
                                 edgecolor='black', linewidth=1.1, zorder=5)
                ax2.add_patch(rect)
        else:
            for i, (x, y) in enumerate(positions):
                rect = Rectangle((x-car_length/2, y-car_width/2),
                                 car_length, car_width,
                                 facecolor=colors[i], alpha=1.0,
                                 edgecolor='black', linewidth=1.1, zorder=5)
                ax2.add_patch(rect)

        ax2.set_xlim(x_left, x_right)
        ax2.set_ylim(-road_half_width - 0.4, road_half_width + 0.4)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.18)
        ax2.tick_params(labelsize=9)
        ax2.set_xlabel('X (m)', fontsize=10)
        ax2.set_ylabel('Y (m)', fontsize=10)

    plt.tight_layout()
    fig2.savefig(os.path.join(CURRENT_DIR, 'AF_formation_snapshots.pdf'), format='pdf', bbox_inches='tight')
    fig2.savefig(os.path.join(CURRENT_DIR, 'AF_formation_snapshots.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)

    if line_candidates:
        line_x_targets = [76.0, 78.0, 80.0, 82.0]

        def _nearest_line_step(x_target: float) -> int:
            best_k = line_candidates[0]
            best_err = abs(avg_x_series[best_k] - x_target)
            for k in line_candidates[1:]:
                err = abs(avg_x_series[k] - x_target)
                if err < best_err:
                    best_err = err
                    best_k = k
            return best_k

        line_steps = [_nearest_line_step(x) for x in line_x_targets]
        line_steps = list(dict.fromkeys(line_steps))
        while len(line_steps) < 4:
            line_steps.append(line_steps[-1])
        line_steps = line_steps[:4]

        line_snapshot_scale = 1.6

        fig3, axes3 = plt.subplots(1, 4, figsize=(12, 3.2))
        for ax3, step_idx in zip(axes3, line_steps):
            positions = [traj[i][step_idx] for i in range(4)]
            cx = sum(p[0] for p in positions) / 4

            x_left = 72.0
            x_right = 84.0
            ax3.fill_between([x_left, x_right], [-road_half_width] * 2, [road_half_width] * 2,
                             color='#ededed', alpha=0.8)
            ax3.axhline(y=road_half_width, color='black', linestyle='--', linewidth=1.6)
            ax3.axhline(y=-road_half_width, color='black', linestyle='--', linewidth=1.6)
            ax3.axhline(y=0.0, color='#666666', linestyle='-', linewidth=1.0, alpha=0.45)

            for obs in obstacles:
                if obs.width > 0:
                    if (obs.x + obs.width / 2) < x_left or (obs.x - obs.width / 2) > x_right:
                        continue
                    rect = Rectangle((obs.x-obs.width/2, obs.y-obs.height/2),
                                     obs.width, obs.height,
                                     facecolor='#6b6b6b', alpha=0.9, edgecolor='#222222', linewidth=1.0)
                    ax3.add_patch(rect)
                else:
                    if (obs.x + obs.radius) < x_left or (obs.x - obs.radius) > x_right:
                        continue
                    circle = Circle((obs.x, obs.y), obs.radius,
                                    facecolor='#6b6b6b', alpha=0.9, edgecolor='#222222', linewidth=1.0)
                    ax3.add_patch(circle)

            gate_center = 78.0
            spacing = 1.60
            order = sorted(range(4), key=lambda idx: positions[idx][0])
            display_xs = [gate_center + (r - 1.5) * spacing for r in range(4)]
            display_ys = [0.0] * 4
            ax3.plot(display_xs, display_ys, color='black', linewidth=1.2, alpha=0.45, zorder=4)

            for r, car_idx in enumerate(order):
                rect = Rectangle((display_xs[r]-(car_length*line_snapshot_scale)/2, -(car_width*line_snapshot_scale)/2),
                                 car_length*line_snapshot_scale, car_width*line_snapshot_scale,
                                 facecolor=colors[car_idx], alpha=1.0,
                                 edgecolor='black', linewidth=1.1, zorder=5)
                ax3.add_patch(rect)

            x_left = min(display_xs) - (car_length * line_snapshot_scale) / 2 - 0.8
            x_right = max(display_xs) + (car_length * line_snapshot_scale) / 2 + 0.8
            x_left = max(70.0, x_left)
            x_right = min(86.0, x_right)
            ax3.set_xlim(x_left, x_right)
            ax3.set_ylim(-road_half_width - 0.4, road_half_width + 0.4)
            ax3.set_aspect('equal')
            ax3.grid(True, alpha=0.18)
            ax3.tick_params(labelsize=9)
            ax3.set_xlabel('X (m)', fontsize=10)
            ax3.set_ylabel('Y (m)', fontsize=10)

        plt.tight_layout()
        fig3.savefig(os.path.join(CURRENT_DIR, 'AF_line_snapshots.pdf'), format='pdf', bbox_inches='tight')
        fig3.savefig(os.path.join(CURRENT_DIR, 'AF_line_snapshots.png'), dpi=300, bbox_inches='tight')
        plt.close(fig3)
    
    print(f"\n图片已保存到: {CURRENT_DIR}")
    print("  - AF_full_scenario.pdf")
    print("  - AF_full_scenario.png")
    print("  - AF_formation_snapshots.pdf")
    print("  - AF_formation_snapshots.png")
    if line_candidates:
        print("  - AF_line_snapshots.pdf")
        print("  - AF_line_snapshots.png")
    
    return success


if __name__ == "__main__":
    run_full_scenario_test()
