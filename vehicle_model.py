"""
车辆模型基础类

包含：
- VehicleState: 车辆状态数据类
- BicycleModel: 自行车运动学模型
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class VehicleState:
    """车辆状态"""
    x: float = 0.0          # x坐标 (m)
    y: float = 0.0          # y坐标 (m)
    theta: float = 0.0      # 航向角 (rad)
    v: float = 0.0          # 速度 (m/s)


class BicycleModel:
    """
    自行车运动学模型
    
    状态: [x, y, theta, v]
    控制: [a, delta] (加速度, 转向角)
    """
    
    def __init__(self, vehicle_params):
        self.L = vehicle_params.L  # 轴距
        self.v_max = vehicle_params.v_max
        self.v_min = vehicle_params.v_min
        self.a_max = vehicle_params.a_max
        self.a_min = vehicle_params.a_min
        self.delta_max = vehicle_params.delta_max
    
    def step(self, state: VehicleState, control: Tuple[float, float], dt: float) -> VehicleState:
        """
        单步状态更新
        
        Args:
            state: 当前状态
            control: (a, delta) 加速度和转向角
            dt: 时间步长
            
        Returns:
            新状态
        """
        x, y, theta, v = state.x, state.y, state.theta, state.v
        a, delta = control
        
        # 限制控制量
        a = np.clip(a, self.a_min, self.a_max)
        delta = np.clip(delta, -self.delta_max, self.delta_max)
        
        # 运动学方程
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + (v / self.L) * np.tan(delta) * dt
        v_new = v + a * dt
        
        # 限制速度
        v_new = np.clip(v_new, self.v_min, self.v_max)
        
        # 归一化角度
        while theta_new > np.pi:
            theta_new -= 2 * np.pi
        while theta_new < -np.pi:
            theta_new += 2 * np.pi
        
        return VehicleState(x=x_new, y=y_new, theta=theta_new, v=v_new)
