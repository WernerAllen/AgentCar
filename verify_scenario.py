"""
验证场景可行性 - 确保合理的编队可以通过所有障碍物
"""
import numpy as np
from config import (
    OBSTACLE_SCENARIOS, CAR_INIT_CONFIGS, 
    FormationParams, EnvParams, VehicleParams
)

def verify_scenario(scenario_name: str = "main"):
    """验证场景中的通道宽度是否足够编队通过"""
    
    obstacles = OBSTACLE_SCENARIOS.get(scenario_name, [])
    formation = FormationParams()
    env = EnvParams()
    vehicle = VehicleParams()
    
    road_hw = env.road_half_width  # 2.5m
    car_radius = vehicle.car_radius  # 0.26m
    
    # 编队宽度（2x2队形，横向间距0.8m）
    # 左车y=0.8, 右车y=-0.8，加上车辆半径，编队宽度约 1.6 + 2*0.26 = 2.12m
    formation_half_width = 0.8 + car_radius  # 1.06m
    
    print(f"=" * 60)
    print(f"场景验证: {scenario_name}")
    print(f"=" * 60)
    print(f"道路半宽: {road_hw}m (总宽{road_hw*2}m)")
    print(f"车辆半径: {car_radius}m")
    print(f"编队半宽: {formation_half_width}m (需通道宽度>{formation_half_width*2:.2f}m)")
    print()
    
    all_passable = True
    
    for i, obs in enumerate(obstacles):
        print(f"--- 障碍物 {i+1}: x={obs.x}m ---")
        
        if obs.width > 0 and obs.height > 0:
            # 矩形障碍物
            half_w = obs.width / 2
            half_h = obs.height / 2
            y_min = obs.y - half_h
            y_max = obs.y + half_h
            x_min = obs.x - half_w
            x_max = obs.x + half_w
            
            print(f"  类型: 矩形 {obs.width}m x {obs.height}m")
            print(f"  位置: 中心({obs.x}, {obs.y})")
            print(f"  范围: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
            
            # 计算可通过的空隙
            # 左侧空隙
            left_gap = road_hw - y_max - car_radius
            # 右侧空隙
            right_gap = y_min + road_hw - car_radius
            # 中间通道（如果障碍物不跨越中线）
            
            print(f"  左侧空隙: {left_gap:.2f}m (道路上边界{road_hw} - 障碍物上边{y_max})")
            print(f"  右侧空隙: {right_gap:.2f}m (障碍物下边{y_min} + 道路下边界{road_hw})")
            
            # 检查是否有足够空间
            min_required = formation_half_width * 2
            if left_gap >= min_required:
                print(f"  -> 左侧可通过 (空隙{left_gap:.2f}m >= 需要{min_required:.2f}m)")
            elif right_gap >= min_required:
                print(f"  -> 右侧可通过 (空隙{right_gap:.2f}m >= 需要{min_required:.2f}m)")
            else:
                # 检查中间通道
                if y_min > 0:  # 障碍物在上侧
                    center_gap = y_min * 2
                elif y_max < 0:  # 障碍物在下侧
                    center_gap = -y_max * 2
                else:
                    center_gap = 0
                
                if center_gap >= min_required:
                    print(f"  -> 中间可通过 (空隙{center_gap:.2f}m)")
                else:
                    # 检查分两边走是否可行（1x2纵向队形）
                    single_lane_width = car_radius * 2  # 单列车辆宽度
                    if left_gap >= single_lane_width and right_gap >= single_lane_width:
                        print(f"  -> 可分两边通过！左{left_gap:.2f}m右{right_gap:.2f}m，各容纳1列车")
                    elif left_gap >= single_lane_width or right_gap >= single_lane_width:
                        print(f"  -> 可单边通过（需队形变换为1x4）")
                    else:
                        print(f"  !! 无法通过! 最大空隙不足")
                        all_passable = False
                    
        else:
            # 圆形障碍物
            print(f"  类型: 圆形, 半径={obs.radius}m")
            print(f"  位置: ({obs.x}, {obs.y})")
            
            # 检查2x2队形是否可直接通过
            # 队形车辆在y=±0.8m，障碍物中心在obs.y
            car_y_positions = [0.8, -0.8]  # 标准队形的y坐标
            can_pass_direct = True
            for car_y in car_y_positions:
                dist_to_obs = abs(car_y - obs.y) - obs.radius
                if dist_to_obs < car_radius:
                    can_pass_direct = False
                    break
            
            if can_pass_direct:
                print(f"  -> 2x2队形可直接通过（车辆在y=±0.8m，避开障碍物）")
            else:
                # 检查两侧空隙
                left_gap = road_hw - obs.y - obs.radius
                right_gap = obs.y + road_hw - obs.radius
                print(f"  左侧空隙: {left_gap:.2f}m, 右侧空隙: {right_gap:.2f}m")
                print(f"  -> 需要队形变换通过")
        
        print()
    
    print(f"=" * 60)
    if all_passable:
        print("结论: 所有障碍物场景理论上可通过")
    else:
        print("结论: 存在无法通过的障碍物!")
    print(f"=" * 60)
    
    return all_passable


def visualize_scenario(scenario_name: str = "main"):
    """可视化场景（ASCII图）"""
    obstacles = OBSTACLE_SCENARIOS.get(scenario_name, [])
    env = EnvParams()
    
    road_hw = env.road_half_width
    
    print(f"\n场景示意图 (y轴向上, x轴向右):")
    print(f"道路范围: y = [{-road_hw}, {road_hw}]")
    
    # 简单ASCII可视化关键障碍物
    for obs in obstacles:
        print(f"\nx={obs.x}m处截面:")
        
        # 创建简单的截面图
        width = 50
        height = 10
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # 道路边界
        for x in range(width):
            grid[0][x] = '-'
            grid[height-1][x] = '-'
        
        # 障碍物位置
        if obs.width > 0 and obs.height > 0:
            y_min = obs.y - obs.height/2
            y_max = obs.y + obs.height/2
            
            # 映射到grid坐标
            grid_y_min = int((road_hw - y_max) / (2*road_hw) * (height-2)) + 1
            grid_y_max = int((road_hw - y_min) / (2*road_hw) * (height-2)) + 1
            
            for y in range(max(1, grid_y_min), min(height-1, grid_y_max+1)):
                for x in range(15, 35):
                    grid[y][x] = '#'
        
        # 打印
        for row in grid:
            print(''.join(row))


def get_expert_action(leader_x: float, obstacles: list) -> np.ndarray:
    """根据障碍物位置返回专家编队动作"""
    action = np.zeros(8)  # [dx0,dy0, dx1,dy1, dx2,dy2, dx3,dy3]
    # 车辆编号：0=前左(y=0.8), 1=前右(y=-0.8), 2=后左(y=0.8), 3=后右(y=-0.8)
    
    for obs in obstacles:
        dist_to_obs = obs.x - leader_x
        
        if 5 < dist_to_obs < 30:  # 接近障碍物
            if obs.y < 0:  # S1: 右侧障碍物 -> 左移
                # 右侧车（1,3）向左移
                action[3] += 0.5   # car1 dy
                action[7] += 0.5   # car3 dy
            elif obs.y > 0:  # S2: 左侧障碍物 -> 右移
                # 左侧车（0,2）向右移
                action[1] -= 0.5   # car0 dy
                action[5] -= 0.5   # car2 dy
            elif obs.y == 0 and obs.width > 2:  # S5: 中间大障碍物 -> 分两边
                action[1] += 0.8   # car0 向左
                action[3] -= 0.8   # car1 向右
                action[5] += 0.8   # car2 向左
                action[7] -= 0.8   # car3 向右
    
    return np.clip(action, -1.0, 1.0)


def test_pure_dwa_run():
    """纯DWA测试（禁用RL）：验证DWA独立避障能力"""
    from formation_rl_env import FormationRLEnv
    import numpy as np
    
    print("\n" + "=" * 60)
    print("纯DWA测试（RL禁用）")
    print("=" * 60)
    
    env = FormationRLEnv(scenario="main", num_cars=4, max_steps=3000, use_rl=False)
    obs, info = env.reset()
    
    total_reward = 0
    done = False
    step = 0
    
    while not done and step < 3000:
        # 零动作，完全依赖DWA
        action = np.zeros(8)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
        
        # 每200步或碰撞时打印
        avg_x = np.mean([car.x for car in env.cars])
        if step % 200 == 0 or done:
            car_ys = [car.y for car in env.cars]
            print(f"Step {step}: x={avg_x:.1f}m, y范围=[{min(car_ys):.2f}, {max(car_ys):.2f}], rl={env.rl_active}")
            if done:
                if terminated:
                    # 诊断碰撞原因
                    for i, car in enumerate(env.cars):
                        print(f"  Car{i}: ({car.x:.1f}, {car.y:.2f})")
                        # 检查边界
                        if abs(car.y) > env.env_params.road_half_width - env.vehicle_params.car_radius:
                            print(f"    -> 边界碰撞!")
                        # 检查障碍物
                        for obs in env.obstacles:
                            if obs.width > 0 and obs.height > 0:
                                half_w, half_h = obs.width/2, obs.height/2
                                dx = max(0, abs(car.x - obs.x) - half_w)
                                dy = max(0, abs(car.y - obs.y) - half_h)
                                if np.sqrt(dx**2 + dy**2) < env.vehicle_params.car_radius:
                                    print(f"    -> 障碍物碰撞! obs@({obs.x}, {obs.y})")
                else:
                    print(f"  -> 超时终止")
    
    env.close()
    print(f"\n最终: {step}步, 总奖励={total_reward:.1f}")
    return step, total_reward


def test_basic_control(scenario: str = "empty", max_steps: int = 500):
    """基础控制测试：验证RL+DWA能否正常运行"""
    from formation_rl_env import FormationRLEnv
    import numpy as np
    
    print(f"\n{'='*60}")
    print(f"基础控制测试: {scenario}场景")
    print(f"{'='*60}")
    
    env = FormationRLEnv(scenario=scenario, num_cars=4, max_steps=max_steps)
    obs, info = env.reset()
    
    done = False
    step = 0
    total_reward = 0
    
    while not done and step < max_steps:
        action = np.zeros(8)  # 零动作
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
    
    avg_x = np.mean([car.x for car in env.cars])
    car_ys = [car.y for car in env.cars]
    
    print(f"结果: {step}步, x={avg_x:.1f}m, y范围=[{min(car_ys):.2f}, {max(car_ys):.2f}]")
    print(f"状态: {'碰撞' if terminated else '正常'}, 奖励={total_reward:.1f}")
    
    env.close()
    return not terminated  # 成功=未碰撞


if __name__ == "__main__":
    # 阶段验证
    print("Stage 1: empty scene")
    if test_basic_control("empty", 500):
        print("[PASS]\n")
        print("Stage 2: s1_right (right obstacle)")
        if test_basic_control("s1_right", 500):
            print("[PASS]\n")
            print("Stage 3: s2_left (left obstacle)")
            test_basic_control("s2_left", 500)
        else:
            print("[FAIL] - need RL training")
    else:
        print("[FAIL] - basic control problem")
