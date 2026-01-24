"""
DWA (Dynamic Window Approach) 控制器
动态窗口法 - 局部路径规划算法

核心思想：
1. 在速度空间中采样可行的(v, w)组合
2. 模拟每个速度对应的轨迹
3. 评估轨迹（目标方向、障碍物距离、速度）
4. 选择最优轨迹对应的速度
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from config import VehicleParams, Obstacle


@dataclass
class DWAParams:
    """DWA参数"""
    # 速度限制
    max_speed: float = 2.0       # 最大线速度 (m/s)
    min_speed: float = 0.8       # 最小线速度 - 强制保持前进（关键修改）
    max_yaw_rate: float = 1.5    # 最大角速度 (rad/s)
    
    # 加速度限制
    max_accel: float = 5.0       # 最大线加速度 - 增大以配合min_speed
    max_yaw_accel: float = 3.0   # 最大角加速度
    
    # 采样参数
    v_resolution: float = 0.1    # 线速度分辨率
    yaw_rate_resolution: float = 0.1  # 角速度分辨率
    
    # 预测参数
    dt: float = 0.1              # 时间步长
    predict_time: float = 2.5    # 预测时间
    
    # 评价函数权重
    heading_weight: float = 2.0   # 朝向目标权重
    dist_weight: float = 1.5      # 障碍物距离权重
    velocity_weight: float = 0.3  # 速度权重
    rl_direction_weight: float = 5.0  # RL方向偏好权重（增大以加快横向响应）
    
    # 安全参数
    robot_radius: float = 0.26   # 机器人半径（与车辆碰撞半径一致）
    safety_margin: float = 0.02  # 安全裕度（极小，只在紧急时避障）


class DWAController:
    """DWA控制器"""
    
    def __init__(self, vehicle_params: VehicleParams = None, params: DWAParams = None):
        self.vehicle_params = vehicle_params or VehicleParams()
        self.params = params or DWAParams()
        
        # 更新机器人半径
        self.params.robot_radius = self.vehicle_params.car_radius
        
        # 道路边界
        self.road_half_width = 2.5
        
        # 调试信息
        self._debug_info = {}
    
    def compute(self, state: Tuple[float, float, float, float],
                goal: Tuple[float, float],
                obstacles: List[Obstacle] = None,
                preferred_y: Optional[float] = None) -> Tuple[float, float]:
        """
        计算控制输入
        
        Args:
            state: (x, y, theta, v) 当前状态
            goal: (gx, gy) 目标位置
            obstacles: 障碍物列表
            preferred_y: RL建议的横向位置（新增）
            
        Returns:
            (acceleration, steering) 控制输入
        """
        x, y, theta, v = state
        obstacles = obstacles or []
        self._preferred_y = preferred_y  # 保存供_calc_cost使用
        
        # 当前角速度（从转向角估计）
        # 简化：假设当前角速度为0
        yaw_rate = 0.0
        
        # 计算动态窗口
        dw = self._calc_dynamic_window(v, yaw_rate)
        
        # ===== 向量化优化：批量采样和评估 =====
        # 生成所有采样点
        v_samples = np.arange(dw[0], dw[1] + self.params.v_resolution, self.params.v_resolution)
        w_samples = np.arange(dw[2], dw[3] + self.params.yaw_rate_resolution, self.params.yaw_rate_resolution)
        
        # 批量预测所有轨迹 (n_v, n_w, n_steps, 3)
        all_trajectories = self._predict_trajectories_batch(x, y, theta, v_samples, w_samples)
        n_v, n_w = len(v_samples), len(w_samples)
        
        # 批量计算代价
        costs = self._calc_costs_batch(all_trajectories, goal, obstacles)
        
        # 添加RL方向偏好代价
        if self._preferred_y is not None:
            end_ys = all_trajectories[:, :, -1, 1]  # (n_v, n_w)
            y_diff = np.abs(end_ys - self._preferred_y)
            rl_cost = self.params.rl_direction_weight * (y_diff / 2.0)
            # 只对非无穷大代价添加
            valid_mask = costs < float('inf')
            costs = np.where(valid_mask, costs + rl_cost, costs)
        
        # 找最优
        best_idx = np.unravel_index(np.argmin(costs), costs.shape)
        best_cost = costs[best_idx]
        best_u = (v_samples[best_idx[0]], w_samples[best_idx[1]]) if best_cost < float('inf') else None
        best_trajectory = all_trajectories[best_idx] if best_u else None
        
        # 如果没有找到可行解，使用简单控制
        if best_u is None:
            self._debug_info = {
                'fallback': True,
                'target_v': None,
                'target_yaw_rate': None,
                'best_cost': float('inf'),
                'trajectory': None
            }
            return self._fallback_control(state, goal)
        
        target_v, target_yaw_rate = best_u
        
        # 转换为加速度和转向角
        acceleration = (target_v - v) / self.params.dt
        acceleration = np.clip(acceleration, -self.params.max_accel, self.params.max_accel)
        
        # 转向角 = 角速度 * 轴距 / 速度 (自行车模型逆运动学)
        if abs(target_v) > 0.1:
            steering = np.arctan(target_yaw_rate * self.vehicle_params.L / target_v)
        else:
            steering = 0.0
        
        steering = np.clip(steering, -self.vehicle_params.delta_max, self.vehicle_params.delta_max)
        
        # 保存调试信息
        self._debug_info = {
            'fallback': False,
            'target_v': target_v,
            'target_yaw_rate': target_yaw_rate,
            'best_cost': best_cost,
            'trajectory': best_trajectory
        }
        
        return acceleration, steering
    
    def _calc_dynamic_window(self, v: float, yaw_rate: float) -> Tuple[float, float, float, float]:
        """计算动态窗口 [v_min, v_max, w_min, w_max]"""
        # 速度窗口（考虑加速度限制）
        v_min = max(self.params.min_speed, v - self.params.max_accel * self.params.dt)
        v_max = min(self.params.max_speed, v + self.params.max_accel * self.params.dt)
        
        # v5.0修复：确保v_min <= v_max（当min_speed动态调整时可能出问题）
        if v_min > v_max:
            v_min = v_max  # 优先保证有可行解
        
        # 角速度窗口
        w_min = max(-self.params.max_yaw_rate, yaw_rate - self.params.max_yaw_accel * self.params.dt)
        w_max = min(self.params.max_yaw_rate, yaw_rate + self.params.max_yaw_accel * self.params.dt)
        
        return (v_min, v_max, w_min, w_max)
    
    def _predict_trajectory(self, x: float, y: float, theta: float,
                           v: float, yaw_rate: float) -> np.ndarray:
        """预测单条轨迹（保留用于调试）"""
        n_steps = int(self.params.predict_time / self.params.dt) + 1
        trajectory = np.zeros((n_steps, 3))
        trajectory[0] = [x, y, theta]
        
        for i in range(1, n_steps):
            theta = trajectory[i-1, 2] + yaw_rate * self.params.dt
            trajectory[i, 0] = trajectory[i-1, 0] + v * np.cos(theta) * self.params.dt
            trajectory[i, 1] = trajectory[i-1, 1] + v * np.sin(theta) * self.params.dt
            trajectory[i, 2] = theta
        
        return trajectory
    
    def _predict_trajectories_batch(self, x: float, y: float, theta: float,
                                    v_samples: np.ndarray, w_samples: np.ndarray) -> np.ndarray:
        """向量化批量预测所有轨迹"""
        n_v, n_w = len(v_samples), len(w_samples)
        n_steps = int(self.params.predict_time / self.params.dt) + 1
        dt = self.params.dt
        
        # 创建网格 (n_v, n_w)
        V, W = np.meshgrid(v_samples, w_samples, indexing='ij')
        
        # 轨迹数组 (n_v, n_w, n_steps, 3) -> [x, y, theta]
        trajectories = np.zeros((n_v, n_w, n_steps, 3))
        trajectories[:, :, 0, 0] = x
        trajectories[:, :, 0, 1] = y
        trajectories[:, :, 0, 2] = theta
        
        # 向量化递推
        for i in range(1, n_steps):
            prev_theta = trajectories[:, :, i-1, 2]
            new_theta = prev_theta + W * dt
            trajectories[:, :, i, 0] = trajectories[:, :, i-1, 0] + V * np.cos(new_theta) * dt
            trajectories[:, :, i, 1] = trajectories[:, :, i-1, 1] + V * np.sin(new_theta) * dt
            trajectories[:, :, i, 2] = new_theta
        
        return trajectories
    
    def _calc_cost(self, trajectory: np.ndarray, goal: Tuple[float, float],
                   obstacles: List[Obstacle]) -> float:
        """计算单条轨迹代价（保留用于调试）"""
        heading_cost = self._calc_heading_cost(trajectory, goal)
        dist_cost = self._calc_obstacle_cost(trajectory, obstacles)
        velocity_cost = self._calc_velocity_cost(trajectory)
        
        if dist_cost == float('inf'):
            return float('inf')
        
        return (self.params.heading_weight * heading_cost +
                self.params.dist_weight * dist_cost +
                self.params.velocity_weight * velocity_cost)
    
    def _calc_costs_batch(self, trajectories: np.ndarray, goal: Tuple[float, float],
                          obstacles: List[Obstacle]) -> np.ndarray:
        """向量化批量计算所有轨迹代价"""
        n_v, n_w, n_steps, _ = trajectories.shape
        
        # 1. 朝向代价 (n_v, n_w)
        end_x = trajectories[:, :, -1, 0]
        end_y = trajectories[:, :, -1, 1]
        end_theta = trajectories[:, :, -1, 2]
        goal_angle = np.arctan2(goal[1] - end_y, goal[0] - end_x)
        angle_diff = np.abs(goal_angle - end_theta)
        heading_cost = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        
        # 2. 障碍物距离代价 (n_v, n_w)
        dist_cost, collision_mask = self._calc_obstacle_cost_batch(trajectories, obstacles)
        
        # 3. 速度代价 (n_v, n_w) - 使用轨迹长度
        dx = np.diff(trajectories[:, :, :, 0], axis=2)
        dy = np.diff(trajectories[:, :, :, 1], axis=2)
        total_dist = np.sum(np.sqrt(dx**2 + dy**2), axis=2)
        expected_dist = self.params.max_speed * self.params.predict_time
        velocity_cost = 1.0 - np.minimum(total_dist / expected_dist, 1.0)
        
        # 综合代价
        total_cost = (self.params.heading_weight * heading_cost +
                      self.params.dist_weight * dist_cost +
                      self.params.velocity_weight * velocity_cost)
        
        # 碰撞轨迹设为无穷大
        total_cost = np.where(collision_mask, float('inf'), total_cost)
        
        return total_cost
    
    def _calc_heading_cost(self, trajectory: np.ndarray, goal: Tuple[float, float]) -> float:
        """计算朝向目标的代价"""
        # 轨迹末端到目标的方向
        end_x, end_y, end_theta = trajectory[-1]
        goal_angle = np.arctan2(goal[1] - end_y, goal[0] - end_x)
        
        # 角度差
        angle_diff = abs(goal_angle - end_theta)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
        
        return angle_diff
    
    def _calc_obstacle_cost(self, trajectory: np.ndarray, obstacles: List[Obstacle]) -> float:
        """计算单条轨迹障碍物代价（保留用于调试）"""
        x_arr = trajectory[:, 0]
        y_arr = trajectory[:, 1]
        
        # 边界检查
        boundary_dist = self.road_half_width - np.abs(y_arr) - self.params.robot_radius
        if np.any(boundary_dist < 0):
            return float('inf')
        min_dist = np.min(boundary_dist)
        
        # 障碍物检查
        for obs in obstacles:
            if obs.width > 0 and obs.height > 0:
                half_w, half_h = obs.width / 2, obs.height / 2
                dx = np.maximum(0, np.abs(x_arr - obs.x) - half_w)
                dy = np.maximum(0, np.abs(y_arr - obs.y) - half_h)
                collision_dist = np.sqrt(dx**2 + dy**2) - self.params.robot_radius - self.params.safety_margin
            else:
                dist = np.sqrt((x_arr - obs.x)**2 + (y_arr - obs.y)**2)
                collision_dist = dist - obs.radius - self.params.robot_radius - self.params.safety_margin
            
            if np.any(collision_dist < 0):
                return float('inf')
            min_dist = min(min_dist, np.min(collision_dist))
        
        # 饱和倒数：避免 dist→0 时代价无界增长
        epsilon = 0.1  # 最大代价 = 1/0.1 = 10
        return 1.0 / (min_dist + epsilon) if min_dist > 0 else 10.0
    
    def _calc_obstacle_cost_batch(self, trajectories: np.ndarray, 
                                   obstacles: List[Obstacle]) -> Tuple[np.ndarray, np.ndarray]:
        """向量化批量计算障碍物代价"""
        n_v, n_w, n_steps, _ = trajectories.shape
        x_all = trajectories[:, :, :, 0]  # (n_v, n_w, n_steps)
        y_all = trajectories[:, :, :, 1]
        
        # 边界检查
        boundary_dist = self.road_half_width - np.abs(y_all) - self.params.robot_radius
        min_dist = np.min(boundary_dist, axis=2)  # (n_v, n_w)
        collision_mask = np.any(boundary_dist < 0, axis=2)
        
        # 障碍物检查
        for obs in obstacles:
            if obs.width > 0 and obs.height > 0:
                half_w, half_h = obs.width / 2, obs.height / 2
                dx = np.maximum(0, np.abs(x_all - obs.x) - half_w)
                dy = np.maximum(0, np.abs(y_all - obs.y) - half_h)
                obs_dist = np.sqrt(dx**2 + dy**2) - self.params.robot_radius - self.params.safety_margin
            else:
                dist = np.sqrt((x_all - obs.x)**2 + (y_all - obs.y)**2)
                obs_dist = dist - obs.radius - self.params.robot_radius - self.params.safety_margin
            
            collision_mask |= np.any(obs_dist < 0, axis=2)
            min_obs_dist = np.min(obs_dist, axis=2)
            min_dist = np.minimum(min_dist, min_obs_dist)
        
        # 饱和倒数：避免 dist→0 时代价无界增长
        epsilon = 0.1  # 最大代价 = 1/0.1 = 10
        dist_cost = np.where(min_dist > 0, 1.0 / (min_dist + epsilon), 10.0)
        
        return dist_cost, collision_mask
    
    def _calc_velocity_cost(self, trajectory: np.ndarray) -> float:
        """计算速度代价（鼓励前进）"""
        # 使用轨迹长度作为速度指标
        if len(trajectory) < 2:
            return 1.0
        
        total_dist = 0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            total_dist += np.sqrt(dx**2 + dy**2)
        
        # 归一化
        expected_dist = self.params.max_speed * self.params.predict_time
        return 1.0 - min(total_dist / expected_dist, 1.0)
    
    def _fallback_control(self, state: Tuple[float, float, float, float],
                         goal: Tuple[float, float]) -> Tuple[float, float]:
        """后备控制（当DWA找不到可行解时）- 保持低速前进"""
        x, y, theta, v = state
        
        # 编队场景：不能停车，保持min_speed前进让编队不散架
        target_v = self.params.min_speed
        if v < target_v:
            return self.params.max_accel * 0.5, 0.0  # 缓慢加速
        elif v > target_v + 0.2:
            return -self.params.max_accel * 0.5, 0.0  # 缓慢减速
        else:
            return 0.0, 0.0  # 保持速度
    
    def get_debug_info(self) -> dict:
        """获取调试信息"""
        return self._debug_info


if __name__ == "__main__":
    # 简单测试
    controller = DWAController()
    
    state = (0.0, 0.0, 0.0, 1.0)  # x, y, theta, v
    goal = (10.0, 0.0)
    obstacles = [Obstacle(x=5.0, y=0.0, radius=0.5)]
    
    a, delta = controller.compute(state, goal, obstacles)
    print(f"Acceleration: {a:.2f}, Steering: {delta:.2f}")
    print(f"Debug: {controller.get_debug_info()}")
