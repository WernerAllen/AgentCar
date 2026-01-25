"""
基于规则的窄门通行方法 (Rule-based Narrow Gate Navigation)

与baseline2的区别：
- Wide Gate (1.6m): 压缩矩形队形通过（不变成纵队）
- Narrow Gate (1.0m): 一字长蛇阵通过

可与RL训练代码无缝衔接。
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 设置中文字体（与baseline2一致）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import Rectangle
from typing import List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import VehicleParams, Obstacle


class RuleBasedPolicy:
    """
    基于规则的策略（可替换RL策略）
    
    输出与RL策略相同格式：4维action，每车一个dy偏移量，范围[-1, 1]
    action=0 表示保持模板位置，action>0 向上偏移，action<0 向下偏移
    
    使用方法：
        policy = RuleBasedPolicy(max_offset=0.5)
        action, _ = policy.predict(obs, car_positions, obstacles)
        # action 可直接传给 FormationRLEnv.step()
    """
    
    def __init__(self, max_offset: float = 0.5):
        self.max_offset = max_offset
        self.lookahead_dist = 30.0  # 前瞻距离（需足够大以提前变阵）
        self.current_mode = "RECT"  # RECT / COMPRESSED / LINE
        
        # 模板位置 (y坐标)
        self.template_y = [0.8, -0.8, 0.8, -0.8]  # Car 0,1,2,3
    
    def predict(self, obs, car_positions, obstacles, deterministic=True):
        """
        生成动作（与 SB3 策略接口兼容）
        
        Args:
            obs: 观测（未使用，仅为兼容RL接口）
            car_positions: 4辆车的 (x, y) 位置
            obstacles: 障碍物列表
            deterministic: 确定性策略
        
        Returns:
            action: shape (4,)，每车一个 dy 偏移，范围 [-1, 1]
            state: None
        """
        avg_x = np.mean([pos[0] for pos in car_positions])
        
        # 检测前方窄门
        gate_type, gate_width = self._detect_narrow_gate(avg_x, obstacles)
        
        # 根据窄门类型决定目标队形
        if gate_type == "narrow":
            target_mode = "LINE"       # 1.0m极窄门 -> 纵队
        elif gate_type == "medium":
            target_mode = "COMPRESSED" # 1.6m窄门 -> 压缩矩形
        else:
            target_mode = "RECT"       # 无窄门 -> 标准队形
        
        # 计算action
        action = self._compute_action(target_mode)
        self.current_mode = target_mode
        
        return action, None
    
    def _detect_narrow_gate(self, avg_x, obstacles):
        """检测前方窄门"""
        road_half_width = 2.5
        upper_bound = road_half_width
        lower_bound = -road_half_width
        
        for obs in obstacles:
            if obs.width > 0:
                obs_x_min = obs.x - obs.width / 2
                obs_x_max = obs.x + obs.width / 2
                
                # 只检测前方障碍物
                if obs_x_max >= avg_x and obs_x_min <= avg_x + self.lookahead_dist:
                    obs_y_min = obs.y - obs.height / 2
                    obs_y_max = obs.y + obs.height / 2
                    
                    if obs.y > 0:
                        upper_bound = min(upper_bound, obs_y_min)
                    else:
                        lower_bound = max(lower_bound, obs_y_max)
        
        gap_width = upper_bound - lower_bound
        
        if gap_width <= 1.2:
            return "narrow", gap_width
        elif gap_width <= 1.8:
            return "medium", gap_width
        else:
            return None, gap_width
    
    def _compute_action(self, target_mode):
        """计算action（与RL格式一致）"""
        action = np.zeros(4, dtype=np.float32)
        
        if target_mode == "RECT":
            # 标准队形，无偏移
            target_y = [0.8, -0.8, 0.8, -0.8]
        elif target_mode == "COMPRESSED":
            # 压缩矩形：保持矩形但缩小y间距
            target_y = [0.35, -0.35, 0.35, -0.35]
        elif target_mode == "LINE":
            # 纵队：所有车向y=0收敛
            target_y = [0.15, -0.15, 0.15, -0.15]
        else:
            target_y = self.template_y
        
        # action = (target_y - template_y) / max_offset
        for i in range(4):
            dy = target_y[i] - self.template_y[i]
            action[i] = np.clip(dy / self.max_offset, -1.0, 1.0)
        
        return action


def run_single_gate_test(
    gate_type: str = "wide",
    max_steps: int = 400,
    dt: float = 0.1,
    road_half_width: float = 2.5,
):
    """
    单窄门测试
    - wide: 1.6m通道 -> 压缩矩形队形
    - narrow: 1.0m通道 -> 一字长蛇阵
    """
    vehicle_params = VehicleParams()
    car_radius = vehicle_params.car_radius
    v_ref = 1.2
    v_fast = v_ref * 1.8
    v_slow = v_ref * 0.3

    # 障碍物配置（与baseline2一致）
    if gate_type == "wide":
        obstacles = [
            Obstacle(20, 1.7, 0.9, "rect", 6.0, 1.8),
            Obstacle(20, -1.7, 0.9, "rect", 6.0, 1.8),
        ]
        gate_name = "Wide Gate"
        init_y = 0.6
        lookahead = 5.0
        # 压缩矩形的目标y（不变成纵队）
        compressed_y = 0.35
    else:
        obstacles = [
            Obstacle(20, 1.5, 1.0, "rect", 6.0, 2.0),
            Obstacle(20, -1.5, 1.0, "rect", 6.0, 2.0),
        ]
        gate_name = "Narrow Gate"
        init_y = 0.4
        lookahead = 8.0
        compressed_y = 0.0  # 一字长蛇

    # 初始化车辆（2x2矩形队形）
    car_states = [
        {'x': 5.0, 'y': init_y, 'v': v_ref},
        {'x': 5.0, 'y': -init_y, 'v': v_ref},
        {'x': 3.0, 'y': init_y, 'v': v_ref},
        {'x': 3.0, 'y': -init_y, 'v': v_ref},
    ]

    goal_x = 35.0
    narrow_zones = [(17, 26)] if gate_type == "narrow" else [(17, 25)]
    
    traj = [[] for _ in range(4)]
    collision_log = []
    current_mode = "RECT"

    print(f"\n{'='*60}")
    print(f"Our Method: {gate_name}")
    print(f"Strategy: {'Compressed Rect' if gate_type == 'wide' else 'Line Formation'}")
    print(f"{'='*60}")

    for step in range(max_steps):
        avg_x = sum(c['x'] for c in car_states) / 4
        
        in_narrow = any(z[0] <= avg_x <= z[1] for z in narrow_zones)
        approaching = any(z[0] - lookahead <= avg_x < z[0] for z in narrow_zones)

        # 状态机
        if current_mode == "RECT" and approaching:
            current_mode = "TO_COMPRESSED" if gate_type == "wide" else "TO_LINE"
            print(f"[Step {step}] x={avg_x:.1f}m: RECT -> {current_mode}")
        
        elif current_mode == "TO_COMPRESSED":
            # 压缩矩形：检查y是否已收敛
            max_y = max(abs(c['y']) for c in car_states)
            if max_y < compressed_y + 0.1:
                current_mode = "COMPRESSED"
                print(f"[Step {step}] x={avg_x:.1f}m: -> COMPRESSED")
        
        elif current_mode == "TO_LINE":
            # 一字长蛇：检查是否已形成纵列
            y_pos_min_x = min(car_states[0]['x'], car_states[2]['x'])
            y_neg_max_x = max(car_states[1]['x'], car_states[3]['x'])
            if y_pos_min_x > y_neg_max_x + 1.5:
                current_mode = "LINE"
                print(f"[Step {step}] x={avg_x:.1f}m: -> LINE")
        
        elif current_mode in ["COMPRESSED", "LINE"] and not in_narrow and not approaching:
            current_mode = "TO_RECT"
            print(f"[Step {step}] x={avg_x:.1f}m: -> TO_RECT")
        
        elif current_mode == "TO_RECT":
            if abs(car_states[0]['x'] - car_states[1]['x']) < 0.5:
                current_mode = "RECT"
                print(f"[Step {step}] x={avg_x:.1f}m: -> RECT")

        # 控制
        for i, car in enumerate(car_states):
            is_y_pos = (i == 0 or i == 2)
            
            if current_mode == "RECT":
                car['v'] = v_ref
                target_y = 0.6 if is_y_pos else -0.6
            
            elif current_mode == "TO_COMPRESSED":
                # 压缩矩形：保持同速，只收缩y
                car['v'] = v_ref
                target_y = compressed_y if is_y_pos else -compressed_y
            
            elif current_mode == "COMPRESSED":
                car['v'] = v_ref
                target_y = compressed_y if is_y_pos else -compressed_y
            
            elif current_mode == "TO_LINE":
                # 一字长蛇：y正的加速，y负的减速
                car['v'] = v_fast if is_y_pos else v_slow
                target_y = 0.15 if is_y_pos else -0.15
            
            elif current_mode == "LINE":
                car['v'] = v_ref
                target_y = 0.0
            
            elif current_mode == "TO_RECT":
                car['v'] = v_slow if is_y_pos else v_fast
                target_y = 0.6 if is_y_pos else -0.6
            
            car['x'] += car['v'] * dt
            car['y'] += (target_y - car['y']) * 0.1

        # 碰撞检测
        for i, car in enumerate(car_states):
            for obs in obstacles:
                ox_min, ox_max = obs.x - obs.width/2, obs.x + obs.width/2
                oy_min, oy_max = obs.y - obs.height/2, obs.y + obs.height/2
                closest_x = max(ox_min, min(car['x'], ox_max))
                closest_y = max(oy_min, min(car['y'], oy_max))
                dist = np.sqrt((car['x']-closest_x)**2 + (car['y']-closest_y)**2)
                if dist < car_radius:
                    collision_log.append((step, avg_x, f"Car{i}-Obs"))

        for i, car in enumerate(car_states):
            traj[i].append((car['x'], car['y']))
        
        if avg_x >= goal_x:
            break

    success = len(collision_log) == 0
    print(f"\nResult: {'PASS' if success else 'FAIL'}")

    # 绘图（与baseline2风格一致）
    fig, ax = plt.subplots(figsize=(8, 2.5))
    car_length = vehicle_params.car_length
    car_width = vehicle_params.car_width
    max_x = max(max(p[0] for p in t) for t in traj if t) + 3

    ax.fill_between([0, max_x], [-road_half_width]*2, [road_half_width]*2, color='lightgray', alpha=0.3)
    ax.axhline(y=road_half_width, color='black', linestyle='--', linewidth=2)
    ax.axhline(y=-road_half_width, color='black', linestyle='--', linewidth=2)

    for obs in obstacles:
        ax.add_patch(Rectangle((obs.x-obs.width/2, obs.y-obs.height/2), obs.width, obs.height,
                                facecolor='gray', alpha=0.7, edgecolor='black'))

    colors = ['red', 'blue', 'green', 'orange']
    
    # 找到窄门处的快照位置（x≈20m）
    snapshot_idx = None
    if traj[0]:
        for idx, (x, y) in enumerate(traj[0]):
            if x >= 20.0:
                snapshot_idx = idx
                break
    
    for i, t in enumerate(traj):
        if t:
            xs, ys = zip(*t)
            # 轨迹用虚线
            ax.plot(xs, ys, color=colors[i], linestyle='--', label=f'Car {i}', linewidth=1.5, alpha=0.7)
            # 起点车辆（实心）
            ax.add_patch(Rectangle((xs[0]-car_length/2, ys[0]-car_width/2), car_length, car_width,
                                    facecolor=colors[i], alpha=0.9, edgecolor='black', zorder=5))
            # 终点车辆（半透明）
            ax.add_patch(Rectangle((xs[-1]-car_length/2, ys[-1]-car_width/2), car_length, car_width,
                                    facecolor=colors[i], alpha=0.5, edgecolor='black', linestyle='--', zorder=5))
            # 窄门处快照（显示压缩队形）
            if snapshot_idx and snapshot_idx < len(t):
                sx, sy = t[snapshot_idx]
                ax.add_patch(Rectangle((sx-car_length/2, sy-car_width/2), car_length, car_width,
                                        facecolor=colors[i], alpha=0.7, edgecolor='black', linewidth=1.5, zorder=6))

    ax.set_xlim(0, max_x + 1)
    ax.set_ylim(-road_half_width - 0.5, road_half_width + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'{gate_name}', fontweight='bold', fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CURRENT_DIR, f'single_gate_{gate_type}.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(CURRENT_DIR, f'single_gate_{gate_type}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return success


if __name__ == "__main__":
    # 测试1: Wide Gate (1.6m) - 压缩矩形
    print("\n" + "="*60)
    print("测试1: Wide Gate (1.6m)")
    print("="*60)
    s1 = run_single_gate_test(gate_type="wide")
    
    # 测试2: Narrow Gate (1.0m) - 一字长蛇
    print("\n" + "="*60)
    print("测试2: Narrow Gate (1.0m)")
    print("="*60)
    s2 = run_single_gate_test(gate_type="narrow")
    
    print("\n" + "="*60)
    print("测试结果汇总:")
    print(f"  Wide Gate (1.6m): {'PASS' if s1 else 'FAIL'}")
    print(f"  Narrow Gate (1.0m): {'PASS' if s2 else 'FAIL'}")
    print("="*60)
    print(f"\nPDF saved to: {CURRENT_DIR}")
