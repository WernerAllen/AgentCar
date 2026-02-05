"""
DWA编队控制系统 - 配置文件
DWA (Dynamic Window Approach) Formation Control Configuration
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

#解释器：D:\Apps-files\anaconda-app\Anaconda_envs\envs\rl_car\python.exe
@dataclass
class VehicleParams:
    """车辆物理参数 - 后轴中心运动学自行车模型"""
    L: float = 0.3              # 轴距 (m)
    car_length: float = 0.45    # 车身长度 (m)
    car_width: float = 0.30     # 车身宽度 (m)
    car_radius: float = 0.27  # 碰撞检测半径 (m) = sqrt((car_length/2)^2 + (car_width/2)^2) ≈ 0.2704
    
    # 速度限制
    v_max: float = 4.0          # 最大前进速度 (m/s)
    v_min: float = -1.0         # 最小速度 (m/s) - 允许慢速后退
    
    # 加速度限制
    a_max: float = 3.0          # 最大加速度 (m/s²)
    a_min: float = -3.0         # 最大减速度 (m/s²)
    
    # 转向限制
    delta_max: float = np.pi / 6  # 最大前轮转角 ±30° (rad)


@dataclass
class MPCParams:
    """MPC 控制器参数"""
    N: int = 18                 # 预测时域步数 (18步，优化计算速度)
    dt: float = 0.15            # 时间步长 (s) - 略增以保持预测距离
    # 预测距离 = N * dt * v_ref ≈ 18 * 0.15 * 2.0 = 5.4m
    # 预测时间 = 18 * 0.15 = 2.7秒
    
    # 代价函数权重 (已调优)
    w_track: float = 8.0        # 目标点追踪权重 (略降，给避障让路)
    w_obs: float = 300.0        # 障碍物避让权重 (增强)
    w_control: float = 0.05     # 控制量权重 (降低，允许更大控制)
    w_smooth: float = 1.0       # 平滑性权重 (增加，减少抖动)
    w_velocity: float = 0.5     # 速度跟踪权重 (降低，避障时允许减速)
    
    # 安全距离
    d_safe_margin: float = 0.12 # 安全余量 (m) - 物理红线
    obs_safe_dist: float = 0.5  # 障碍物安全距离 (m) - 增大以留更多余量
    car_min_dist: float = 0.6   # 车间最小距离 (m)
    
    # 参考速度
    v_ref: float = 2.0          # 参考前进速度 (m/s) - 略降，更稳定


@dataclass
class FormationParams:
    """编队参数"""
    # 标准编队间距
    d_lateral: float = 0.8      # 横向间距 (m)
    d_longitudinal: float = 1.2 # 纵向间距 (m)
    
    # 编队模板偏移 (相对于虚拟领航点) - 2x2正方形
    template_offsets: List[Tuple[float, float]] = field(default_factory=lambda: [
        (1.0, 0.8),    # Car 0: 前左
        (1.0, -0.8),   # Car 1: 前右
        (-1.0, 0.8),   # Car 2: 后左
        (-1.0, -0.8),  # Car 3: 后右
    ])


@dataclass
class EnvParams:
    """仿真环境参数"""
    road_half_width: float = 2.5   # 道路半宽 (m)
    goal_x: float = 150.0          # 终点 x 坐标 (m) - 覆盖所有障碍物场景
    
    # 虚拟领航点
    leader_speed: float = 1.5      # 虚拟领航点速度 (m/s)
    leader_start_x: float = 3.2    # 虚拟领航点起始x - 与2x2车队中心对齐
    
    # 仿真步长
    sim_dt: float = 0.1            # 仿真步长 (s)
    
    # 渲染
    render_fps: int = 30


@dataclass
class Obstacle:
    """静态障碍物 - 支持多种类型"""
    x: float
    y: float
    radius: float  # 等效半径（用于碰撞检测）
    obs_type: str = "circle"  # 类型: circle(圆形), rect(矩形), cone(锥桶), person(行人), car(车辆)
    width: float = 0.0   # 矩形宽度 (仅rect类型)
    height: float = 0.0  # 矩形高度 (仅rect类型)
    color: str = ""      # 自定义颜色 (可选)


# 预设障碍物场景
# 道路半宽2.5m，车辆编队宽度约1.6m
OBSTACLE_SCENARIOS = {
    # ========== 主测试场景: 5种障碍物类型 ==========
    # 道路半宽2.5m，编队宽度约1.6m
    # 障碍物与边界连接，模拟真实路况
    "main": [
        # ===== 空白段 (0-20m): 起步区 =====
        
        # ===== 场景1 (25m): 右侧施工区，从左通过 =====
        # 障碍物高度2m，贴右边界
        Obstacle(25, -1.5, 1.0, "rect", 6.0, 2.0),   # 右侧 y=-2.5到y=-0.5
        
        # ===== 空白段 (35-45m) =====
        
        # ===== 场景2 (50m): 左侧停车区，从右通过 =====
        # 障碍物高度2m，贴左边界
        Obstacle(50, 1.5, 1.0, "rect", 6.0, 2.0),    # 左侧 y=0.5到y=2.5
        
        # ===== 空白段 (60-70m) =====
        
        # ===== 场景3 (75m): 两侧施工，中间通道 =====
        # 两侧各1.2m高障碍物
        Obstacle(75, 1.9, 0.6, "rect", 6.0, 1.2),    # 左侧 y=1.3到y=2.5
        Obstacle(75, -1.9, 0.6, "rect", 6.0, 1.2),   # 右侧 y=-2.5到y=-1.3
        
        # ===== 空白段 (85-95m) =====
        
        # ===== 场景4 (100m): 中间小障碍，两侧通过 =====
        Obstacle(100, 0.0, 0.4, "circle"),           # 中间小障碍
        
        # ===== 空白段 (105-120m) =====
        
        # ===== 场景5 (125m): 中间大障碍，两边空隙通行 =====
        # 中间2.6m高障碍物，两边各留1.2m空隙
        Obstacle(125, 0.0, 1.3, "car", 6.0, 2.6),    # 中间 y=-1.3到y=1.3，两边各1.2m空隙
        
        # ===== 空白段 (130-150m): 终点区 =====
    ],
    
    # 空场景（调试用）
    "empty": [],
    
    # ========== 单独场景（课程学习）==========
    # 每个场景包含多组障碍物，间隔约30m，总长约100m
    
    # S1: 右侧障碍x3 -> 练习左移通过
    "s1_right": [
        Obstacle(25, -1.5, 1.0, "rect", 6.0, 2.0),
        Obstacle(55, -1.5, 1.0, "rect", 6.0, 2.0),
        Obstacle(85, -1.5, 1.0, "rect", 6.0, 2.0),
    ],
    
    # S2: 左侧障碍x3 -> 练习右移通过
    "s2_left": [
        Obstacle(25, 1.5, 1.0, "rect", 6.0, 2.0),
        Obstacle(55, 1.5, 1.0, "rect", 6.0, 2.0),
        Obstacle(85, 1.5, 1.0, "rect", 6.0, 2.0),
    ],
    
    # S1S2混合: 左右交替障碍 -> 学会根据障碍位置决定避障方向（解决负迁移问题）
    "s1s2_mixed": [
        Obstacle(25, -1.5, 1.0, "rect", 6.0, 2.0),   # 右侧 -> 左移
        Obstacle(55, 1.5, 1.0, "rect", 6.0, 2.0),    # 左侧 -> 右移
        Obstacle(85, -1.5, 1.0, "rect", 6.0, 2.0),   # 右侧 -> 左移
    ],
    
    # S3: 两侧障碍x3 -> 练习窄道通过，保持原队形无法通过必须压缩队形
    # 编队宽度1.9m(中心1.6m+车宽0.3m)，通道设为1.6m，强制横向压缩
    # 障碍物: y=±1.7, height=1.8 -> 占据 y=[0.8,2.5]和[-2.5,-0.8] -> 通道=[-0.8,0.8]=1.6m
    # 减少障碍物长度(6m->3m)，给自行车模型更多变道空间
    "s3_narrow": [
        Obstacle(25, 1.7, 0.9, "rect", 3.0, 1.8),
        Obstacle(25, -1.7, 0.9, "rect", 3.0, 1.8),
        Obstacle(55, 1.7, 0.9, "rect", 3.0, 1.8),
        Obstacle(55, -1.7, 0.9, "rect", 3.0, 1.8),
        Obstacle(85, 1.7, 0.9, "rect", 3.0, 1.8),
        Obstacle(85, -1.7, 0.9, "rect", 3.0, 1.8),
    ],
    
    # S4: 中间小障碍x3 -> 入门避障（半径0.25m，队形不变可通过）
    "s4_center_small": [
        Obstacle(25, 0.0, 0.25, "circle"),
        Obstacle(55, 0.0, 0.25, "circle"),
        Obstacle(85, 0.0, 0.25, "circle"),
    ],
    
    # S5: 中间大障碍x2 -> 分两边通过
    "s5_center_large": [
        Obstacle(30, 0.0, 1.3, "car", 6.0, 2.6),
        Obstacle(70, 0.0, 1.3, "car", 6.0, 2.6),
    ],
    
    # S1-S6全场景综合: 右侧 -> 左侧 -> 窄门 -> 小障 -> 大障 -> 极窄
    "comprehensive_all": [
        # 场景1: 右侧障碍 (x=15m) -> 左移避让
        Obstacle(15, -1.5, 1.0, "rect", 3.2, 2.0),
        
        # 场景2: 左侧障碍 (x=26m) -> 右移避让
        Obstacle(26, 1.5, 1.0, "rect", 3.2, 2.0),
        
        # 场景3: 两侧窄门 (x=37m, 1.6m宽) -> 压缩矩形队形
        Obstacle(37, 1.7, 0.9, "rect", 3.2, 1.8),
        Obstacle(37, -1.7, 0.9, "rect", 3.2, 1.8),
        
        # 场景4: 中间小障碍 (x=48m) -> 微调避让
        Obstacle(48, 0.0, 0.4, "circle"),
        
        # 场景5: 中间大障碍 (x=59m) -> 分流
        Obstacle(59, 0.0, 1.3, "car", 3.5, 2.6),
        
        # 场景6: 极窄通道 (x=78m, 1.0m宽) -> 一字长蛇阵
        Obstacle(78, 1.5, 1.0, "rect", 4.0, 2.0),
        Obstacle(78, -1.5, 1.0, "rect", 4.0, 2.0),
    ],
    
    # S6: 极窄通道x2 -> 必须变成1×4纵队（一字长蛇阵）通过
    # 减少障碍物长度(10m->5m)，给纵队变阵更多空间
    "s6_very_narrow": [
        # 第一组极窄通道 (x=30m)
        # 通道宽度=1.0m: y ∈ [-0.5, 0.5]
        Obstacle(30, 1.5, 1.0, "rect", 5.0, 2.0),   # 上侧障碍物，占y=[0.5, 2.5]
        Obstacle(30, -1.5, 1.0, "rect", 5.0, 2.0),  # 下侧障碍物，占y=[-2.5, -0.5]
        # 第二组极窄通道 (x=70m)
        Obstacle(70, 1.5, 1.0, "rect", 5.0, 2.0),
        Obstacle(70, -1.5, 1.0, "rect", 5.0, 2.0),
    ],
    
    # ========== 基线算法测试场景 ==========
    # 双窄门测试：S3窄门(1.6m) + S6极窄门(1.0m)
    "double_narrow": [
        # 第一道窄门 (x=25m): S3配置，通道1.6m
        # 编队需从1.9m压缩到1.6m
        Obstacle(25, 1.7, 0.9, "rect", 6.0, 1.8),    # 上侧，占y=[0.8, 2.5]
        Obstacle(25, -1.7, 0.9, "rect", 6.0, 1.8),   # 下侧，占y=[-2.5, -0.8]
        # 第二道极窄门 (x=55m): S6配置，通道1.0m
        # 编队需变成纵队通过
        Obstacle(55, 1.5, 1.0, "rect", 10.0, 2.0),   # 上侧，占y=[0.5, 2.5]
        Obstacle(55, -1.5, 1.0, "rect", 10.0, 2.0),  # 下侧，占y=[-2.5, -0.5]
    ],
    
}


# 小车初始配置 - 2x2队形（与template_offsets一致）
CAR_INIT_CONFIGS = [
    {"name": "car_0", "color": "red",    "init_pos": (5.0, 0.8)},   # 前左
    {"name": "car_1", "color": "blue",   "init_pos": (5.0, -0.8)},  # 前右
    {"name": "car_2", "color": "green",  "init_pos": (3.0, 0.8)},   # 后左
    {"name": "car_3", "color": "yellow", "init_pos": (3.0, -0.8)},  # 后右
]


def get_default_config():
    """获取默认配置"""
    return {
        "vehicle": VehicleParams(),
        "mpc": MPCParams(),
        "formation": FormationParams(),
        "env": EnvParams(),
    }
