"""
DWA编队环境 - 无强化学习版本
使用DWA(Dynamic Window Approach)进行避障控制
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from config import (
    VehicleParams, FormationParams, EnvParams,
    Obstacle, OBSTACLE_SCENARIOS, CAR_INIT_CONFIGS, get_default_config
)
from vehicle_model import VehicleState, BicycleModel
from dwa_controller import DWAController, DWAParams


@dataclass
class CarState:
    """单车状态"""
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    
    def to_vehicle_state(self) -> VehicleState:
        return VehicleState(x=self.x, y=self.y, theta=self.theta, v=self.v)


class VirtualLeader:
    """虚拟领航点"""
    
    def __init__(self, start_x: float, speed: float):
        self.x = start_x
        self.y = 0.0
        self.base_speed = speed
        self.speed = speed
        self.max_lag_tolerance = 3.0
    
    def update(self, dt: float, car_states: List[CarState] = None):
        """更新领航者位置"""
        speed_ratio = 1.0
        
        if car_states and len(car_states) > 0:
            # 位置落后反馈
            min_car_x = min(car.x for car in car_states)
            lag = self.x - min_car_x
            if lag > self.max_lag_tolerance:
                lag_ratio = max(0.2, 1.0 - (lag - self.max_lag_tolerance) / 5.0)
                speed_ratio = min(speed_ratio, lag_ratio)
        
        self.speed = self.base_speed * speed_ratio
        self.x += self.speed * dt
    
    def get_formation_targets(self, offsets: List[Tuple[float, float]],
                               scale_x: float = 1.0, scale_y: float = 1.0) -> List[Tuple[float, float]]:
        """计算各车的目标位置"""
        return [(self.x + dx * scale_x, self.y + dy * scale_y) for dx, dy in offsets]
    
    def reset(self, start_x: float):
        self.x = start_x
        self.y = 0.0
        self.speed = self.base_speed


class DWAFormationEnv:
    """
    DWA编队环境 - 纯控制版本（无RL）
    
    使用DWA算法进行局部避障和路径规划
    """
    
    def __init__(self,
                 scenario: str = "simple",
                 num_cars: int = 4,
                 config: Optional[Dict] = None):
        
        # 加载配置
        self.config = config or get_default_config()
        self.vehicle_params: VehicleParams = self.config["vehicle"]
        self.formation_params: FormationParams = self.config["formation"]
        self.env_params: EnvParams = self.config["env"]
        
        self.num_cars = num_cars
        self.scenario = scenario
        
        # 加载障碍物
        self.obstacles = OBSTACLE_SCENARIOS.get(scenario, [])
        
        # 初始化组件
        self.bicycle_model = BicycleModel(self.vehicle_params)
        self.virtual_leader = VirtualLeader(
            self.env_params.leader_start_x,
            self.env_params.leader_speed
        )
        
        # DWA控制器（每车一个）
        self.dwa_controllers: List[DWAController] = []
        for _ in range(num_cars):
            dwa = DWAController(vehicle_params=self.vehicle_params)
            dwa.road_half_width = self.env_params.road_half_width
            self.dwa_controllers.append(dwa)
        
        # 状态变量
        self.car_states: List[CarState] = []
        self.step_count = 0
        self.max_steps = 2000
        
        # 渲染
        self.fig = None
        self.ax = None
    
    def reset(self) -> Dict:
        """重置环境"""
        # 重置虚拟领航点
        self.virtual_leader.reset(self.env_params.leader_start_x)
        
        # 重置车辆状态
        self.car_states = []
        for i in range(self.num_cars):
            cfg = CAR_INIT_CONFIGS[i]
            init_v = 1.0
            state = CarState(
                x=cfg["init_pos"][0],
                y=cfg["init_pos"][1],
                vx=init_v, vy=0.0,
                theta=0.0,
                v=init_v
            )
            self.car_states.append(state)
        
        self.step_count = 0
        
        return self._get_info()
    
    def step(self) -> Tuple[bool, Dict]:
        """执行一步仿真"""
        self.step_count += 1
        
        # 计算变形矩阵（队形压缩）
        scale_y = 1.0
        leader_x = self.virtual_leader.x
        
        for obs in self.obstacles:
            dx = obs.x - leader_x
            if 0 < dx < 10:
                lateral_clearance = self.env_params.road_half_width - abs(obs.y) - obs.radius
                if lateral_clearance < 1.5:
                    compress_ratio = max(0.5, lateral_clearance / 1.5)
                    scale_y = min(scale_y, compress_ratio)
        
        # 计算各车的目标位置
        template_targets = self.virtual_leader.get_formation_targets(
            self.formation_params.template_offsets[:self.num_cars],
            scale_x=1.0,
            scale_y=scale_y
        )
        
        # 控制每辆车
        for i, (car_state, target) in enumerate(zip(self.car_states, template_targets)):
            # 获取附近障碍物（增大检测范围）
            nearby_obs = self._get_nearby_obstacles(car_state, lookahead=20.0)
            
            # 添加其他车辆作为动态障碍物（增大安全裕度）
            for j, other_car in enumerate(self.car_states):
                if j != i:
                    dx = other_car.x - car_state.x
                    dy = other_car.y - car_state.y
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < 5.0:  # 增大检测范围
                        car_obs = Obstacle(
                            x=other_car.x,
                            y=other_car.y,
                            radius=self.vehicle_params.car_radius + 0.5,  # 增大安全裕度
                            obs_type="car"
                        )
                        nearby_obs.append(car_obs)
            
            # DWA控制 - 使用前方远点作为目标（避免局部最优）
            state = (car_state.x, car_state.y, car_state.theta, car_state.v)
            # 目标点：前方10m + 编队横向偏移
            lookahead_goal = (car_state.x + 10.0, target[1])
            a, delta = self.dwa_controllers[i].compute(state, lookahead_goal, nearby_obs)
            
            # 更新状态
            new_state = self.bicycle_model.step(
                car_state.to_vehicle_state(),
                (a, delta),
                self.env_params.sim_dt
            )
            
            car_state.x = new_state.x
            car_state.y = new_state.y
            car_state.theta = new_state.theta
            car_state.v = new_state.v
            car_state.vx = new_state.v * np.cos(new_state.theta)
            car_state.vy = new_state.v * np.sin(new_state.theta)
        
        # 更新虚拟领航点
        self.virtual_leader.update(self.env_params.sim_dt, self.car_states)
        
        # 检查终止条件
        done, info = self._check_termination()
        
        return done, info
    
    def _get_nearby_obstacles(self, car: CarState, lookahead: float = 10.0) -> List[Obstacle]:
        """获取车辆前方的障碍物"""
        nearby = []
        for obs in self.obstacles:
            dx = obs.x - car.x
            if -2.0 < dx < lookahead:
                nearby.append(obs)
        return nearby
    
    def _check_termination(self) -> Tuple[bool, Dict]:
        """检查终止条件"""
        info = self._get_info()
        
        for i, car in enumerate(self.car_states):
            # 到达终点
            if i == 0 and car.x >= self.env_params.goal_x:
                info["success"] = True
                info["message"] = "Goal reached!"
                return True, info
            
            # 障碍物碰撞
            for obs in self.obstacles:
                dist = np.sqrt((car.x - obs.x)**2 + (car.y - obs.y)**2)
                if dist < obs.radius + self.vehicle_params.car_radius:
                    info["collision"] = f"obstacle (car_{i})"
                    info["message"] = f"Car {i} hit obstacle!"
                    return True, info
            
            # 边界碰撞
            if abs(car.y) > self.env_params.road_half_width - self.vehicle_params.car_radius:
                info["collision"] = f"boundary (car_{i})"
                info["message"] = f"Car {i} hit boundary!"
                return True, info
        
        # 车间碰撞
        for i in range(len(self.car_states)):
            for j in range(i + 1, len(self.car_states)):
                car_i = self.car_states[i]
                car_j = self.car_states[j]
                dist = np.sqrt((car_i.x - car_j.x)**2 + (car_i.y - car_j.y)**2)
                if dist < 2 * self.vehicle_params.car_radius:
                    info["collision"] = f"car_{i} <-> car_{j}"
                    info["message"] = f"Car {i} and Car {j} collided!"
                    return True, info
        
        # 超时
        if self.step_count >= self.max_steps:
            info["timeout"] = True
            info["message"] = "Timeout!"
            return True, info
        
        return False, info
    
    def _get_info(self) -> Dict[str, Any]:
        """获取状态信息"""
        if not self.car_states:
            return {"step": 0}
        
        car = self.car_states[0]
        return {
            "step": self.step_count,
            "leader_x": self.virtual_leader.x,
            "cars": [(c.x, c.y, c.v) for c in self.car_states]
        }
    
    def render(self):
        """渲染环境"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(14, 5))
        
        self.ax.clear()
        
        # 道路边界
        self.ax.axhline(y=self.env_params.road_half_width, color='gray', linestyle='--', linewidth=2)
        self.ax.axhline(y=-self.env_params.road_half_width, color='gray', linestyle='--', linewidth=2)
        
        # 计算视窗
        if self.car_states:
            center_x = self.car_states[0].x
        else:
            center_x = 0
        
        # 障碍物
        for obs in self.obstacles:
            if center_x - 5 < obs.x < center_x + 25:
                circle = patches.Circle((obs.x, obs.y), obs.radius, 
                                        color='darkgray', alpha=0.8)
                self.ax.add_patch(circle)
        
        # 车辆
        colors = ['red', 'blue', 'green', 'orange']
        for i, car in enumerate(self.car_states):
            color = colors[i % len(colors)]
            car_circle = patches.Circle((car.x, car.y), self.vehicle_params.car_radius,
                                        color=color, alpha=0.8)
            self.ax.add_patch(car_circle)
            
            # 速度方向
            self.ax.arrow(car.x, car.y, 
                         car.vx * 0.3, car.vy * 0.3,
                         head_width=0.1, color=color)
        
        # 虚拟领航点
        self.ax.scatter(self.virtual_leader.x, self.virtual_leader.y,
                       color='purple', marker='x', s=150, linewidths=3, label='Leader')
        
        # 目标位置
        targets = self.virtual_leader.get_formation_targets(
            self.formation_params.template_offsets[:self.num_cars]
        )
        for i, (tx, ty) in enumerate(targets):
            self.ax.scatter(tx, ty, color=colors[i % len(colors)], 
                          marker='+', s=100, alpha=0.5)
        
        self.ax.set_xlim(center_x - 5, center_x + 25)
        self.ax.set_ylim(-4, 4)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title(f'DWA Formation Control | Step: {self.step_count} | '
                         f'Leader: {self.virtual_leader.x:.1f}m')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        
        plt.pause(0.01)
    
    def close(self):
        """关闭环境"""
        import matplotlib.pyplot as plt
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


def run_simulation(scenario: str = "simple", num_cars: int = 4, render: bool = True):
    """运行仿真"""
    env = DWAFormationEnv(scenario=scenario, num_cars=num_cars)
    env.reset()
    
    done = False
    while not done:
        done, info = env.step()
        
        if render:
            env.render()
        
        if done:
            print(f"\n=== Simulation Finished ===")
            print(f"Steps: {info['step']}")
            if "success" in info:
                print(f"Result: SUCCESS!")
            elif "collision" in info:
                print(f"Result: COLLISION - {info['collision']}")
            elif "timeout" in info:
                print(f"Result: TIMEOUT")
            break
    
    if render:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()
    
    env.close()
    return info


if __name__ == "__main__":
    run_simulation(scenario="simple", num_cars=4, render=True)
