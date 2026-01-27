"""
集中式RL编队控制环境
- 状态空间：Dict观测（图像4×84×84 + 向量72维=3帧×24维）
- 动作空间：4维（4车dy修正；兼容8维输入但会忽略dx并抽取dy）
- 执行层：DWA控制器
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Optional, List
from collections import deque
from dataclasses import dataclass

from config import (
    OBSTACLE_SCENARIOS, VehicleParams, EnvParams, 
    FormationParams, CAR_INIT_CONFIGS, Obstacle
)
from dwa_controller import DWAController, DWAParams
from vehicle_model import BicycleModel, VehicleState


@dataclass
class GridMapConfig:
    """栅格图配置"""
    size: int = 84                # 栅格图尺寸
    resolution: float = 0.25      # 每格分辨率 (m)
    range_x: float = 10.5         # x方向范围 (m) = 84 * 0.25 / 2
    range_y: float = 10.5         # y方向范围 (m)


class FormationRLEnv(gym.Env):
    """
    集中式RL编队控制环境
    
    上层RL Agent：输出4车目标点修正量
    下层DWA：执行轨迹跟踪和避障
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        scenario: str = "main",
        num_cars: int = 4,
        max_steps: int = 2000,
        use_rl: bool = True,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.scenario = scenario
        self.num_cars = num_cars
        self.max_steps = max_steps
        self.use_rl = use_rl
        self.render_mode = render_mode
        
        # 参数
        self.vehicle_params = VehicleParams()
        self.env_params = EnvParams()
        self.formation_params = FormationParams()
        self.grid_config = GridMapConfig()
        
        # 障碍物
        self.obstacles = OBSTACLE_SCENARIOS.get(scenario, [])
        
        # DWA控制器（每车一个）
        self.dwa_params = DWAParams()
        self.dwa_controllers = [
            DWAController(self.vehicle_params, self.dwa_params) 
            for _ in range(num_cars)
        ]
        
        # 车辆模型
        self.bicycle = BicycleModel(self.vehicle_params)
        
        # ========== 状态空间 ==========
        # 使用Dict空间，包含图像和向量特征
        # 图像: 4通道栅格图 (4, 84, 84)
        # 向量: 堆叠3帧状态，每帧24维 = 72维
        #   每帧: delta_y(4) + 车辆y(4) + 速度(4) + 理想y(4) + 障碍物距离(4) + 障碍物方向(4) = 24维
        self.frame_stack_size = 3  # 帧堆叠数量
        self.vector_dim_per_frame = 24  # 4(delta_y) + 4(car_y) + 4(v) + 4(ideal_y) + 8(障碍物) = 24
        
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0.0, high=1.0,
                shape=(4, self.grid_config.size, self.grid_config.size),
                dtype=np.float32
            ),
            "vector": spaces.Box(
                low=-10.0, high=10.0,
                shape=(self.frame_stack_size * self.vector_dim_per_frame,),
                dtype=np.float32
            )
        })
        
        # 帧堆叠缓冲区（使用deque优化O(1)操作）
        self.vector_history = deque(maxlen=self.frame_stack_size)
        
        # ========== 动作空间 ==========
        # 4维：每车dy增量，范围[-1.0, 1.0]
        # 简化设计：dx无效（DWA目标x固定为car.x+lookahead）
        # 累积后总修正范围[-2, 2]米
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(4,),  # 只有dy
            dtype=np.float32
        )
        
        # 内部状态
        self.cars: List[VehicleState] = []
        self.step_count = 0
        self.leader_x = 0.0  # 虚拟领航点x坐标
        self.rl_active = False  # RL是否激活

        # 诊断缓存（用于训练日志）
        self._last_passage_width = self.env_params.road_half_width * 2
        self._last_passage_center = 0.0
        self._last_scale_factor = 1.0
        self._last_target_adjustment_mean = 0.0
        self._last_target_adjustment_max = 0.0
        self._last_target_adjustment_count = 0
        self._last_dwa_fallback_count = 0
        
        # 累积修正量（只有dy，4维）
        self.delta_accumulator = np.zeros(4, dtype=np.float32)  # 4车的dy
        self.smooth_factor = 0.3  # 激活时平滑因子
        self.decay_factor = 0.9   # 休眠时衰减因子
        
        # 时间抽象：降低RL决策频率（舰长vs舵手设计）
        # RL每decision_interval步才真正决策一次，中间保持上次action
        self.decision_interval = 5  # 每5步决策一次（0.5秒@10Hz）
        self.last_action = np.zeros(4, dtype=np.float32)
        
        # 碰撞预测距离
        self.collision_check_dist = 20.0  # 前方20m内检测（提前触发）

        # 通道检测前视距离（过早触发会导致提前收缩/碰撞）
        self.narrow_lookahead = 8.0
        self.very_narrow_lookahead = 10.0
        self.very_narrow_exit_margin = 6.0
        self.very_narrow_recover_dist = 8.0
        # 前车追尾缓冲距离（仅在收缩/极窄模式启用）
        self.front_buffer_dist = 1.6
        self.follow_brake_dist = 2.8

        # 极窄通道状态（用于跨门保持纵队）
        self._very_narrow_active = False
        self._very_narrow_seen = False
        self._very_narrow_exit_x = None
        self._very_narrow_min_width = float('inf')
        self._last_is_very_narrow = False
        self._last_very_narrow_width = float('inf')
        self._very_narrow_recover_alpha = 1.0
        self._build_very_narrow_segments()
        self._row_groups = self._build_row_groups()
        
        # 轨迹记录
        self.trajectories = [[] for _ in range(num_cars)]
        
        # ===== 性能优化：缓存机制 =====
        self._cache_step = -1  # 缓存有效的step
        self._cached_ideal_positions = None
        self._cached_passage_bounds = {}  # key: lookahead, value: (upper, lower)
    
    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        
        # 重置车辆位置（固定理想位置，预训练为零action不需要随机扰动）
        self.cars = []
        for i in range(self.num_cars):
            cfg = CAR_INIT_CONFIGS[i]
            x, y = cfg['init_pos']
            self.cars.append(VehicleState(x=x, y=y, theta=0.0, v=0.5))
        
        # 重置领航点
        self.leader_x = self.env_params.leader_start_x
        
        # 重置计数器
        self.step_count = 0
        self.rl_active = self.use_rl

        # 重置诊断缓存
        self._last_passage_width = self.env_params.road_half_width * 2
        self._last_passage_center = 0.0
        self._last_scale_factor = 1.0
        self._last_target_adjustment_mean = 0.0
        self._last_target_adjustment_max = 0.0
        self._last_target_adjustment_count = 0
        self._last_dwa_fallback_count = 0
        self._very_narrow_active = False
        self._very_narrow_seen = False
        self._last_is_very_narrow = False
        self._last_very_narrow_width = float('inf')
        self._very_narrow_recover_alpha = 1.0
        self._row_groups = self._build_row_groups()
        
        # 重置累积修正量（4维dy）
        self.delta_accumulator = np.zeros(4, dtype=np.float32)
        self.last_action = np.zeros(4, dtype=np.float32)  # 重置时间抽象的缓存
        
        # 清空轨迹
        self.trajectories = [[(car.x, car.y)] for car in self.cars]
        self._last_avg_x = np.mean([car.x for car in self.cars])
        
        # 重置帧堆叠缓冲区（用当前帧填充）
        current_vector = self._get_current_vector()
        self.vector_history = deque([current_vector.copy() for _ in range(self.frame_stack_size)], maxlen=self.frame_stack_size)
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        执行一步
        
        action: shape (8,) - 4车目标点修正量 [dx0, dy0, dx1, dy1, dx2, dy2, dx3, dy3]
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] == 8:
            action = action[[1, 3, 5, 7]]
        elif action.shape[0] != 4:
            raise ValueError(
                f"Invalid action shape {action.shape}. Expected (4,) or (8,) for {self.num_cars} cars."
            )
        self.step_count += 1
        
        # 更新领航点
        self.leader_x += self.env_params.leader_speed * self.env_params.sim_dt
        
        # 计算理想队形位置
        ideal_positions = self._compute_ideal_positions()
        
        # 时间抽象：RL每decision_interval步才决策一次
        # 中间步骤保持上次action，避免微操过头（Jittery）
        self.rl_active = self.use_rl
        if self.use_rl:
            if self.step_count % self.decision_interval == 1:
                # 决策时刻：更新action
                self.last_action = action.copy()
            # 优先级4修复：自适应平滑因子
            # 窄道时需要更快响应(α=0.5)，宽道时更平滑(α=0.3)
            avg_x = np.mean([car.x for car in self.cars])
            upper_b, lower_b = self._get_passage_bounds(avg_x, lookahead=self.narrow_lookahead)
            pw = upper_b - lower_b
            adaptive_alpha = 0.5 if pw < 2.1 else self.smooth_factor  # 窄道时α更大
            self.delta_accumulator = (1 - adaptive_alpha) * self.delta_accumulator + \
                                     adaptive_alpha * self.last_action
        else:
            self.delta_accumulator = self.delta_accumulator * self.decay_factor
        self.delta_accumulator = np.clip(self.delta_accumulator, -2.0, 2.0)
        
        # 应用偏移（只有dy，dx保持0）
        deltas = np.zeros((self.num_cars, 2), dtype=np.float32)
        deltas[:, 1] = self.delta_accumulator
        target_positions = ideal_positions + deltas
        raw_target_positions = target_positions.copy()
        
        # 关键安全约束：防止RL输出导致目标位置太近
        # 确保同排两车目标y间距 >= 碰撞阈值
        self._enforce_row_spacing(target_positions)

        # 记录目标点被安全约束修正的幅度（诊断用）
        target_adjustments = np.abs(target_positions[:, 1] - raw_target_positions[:, 1])
        self._last_target_adjustment_mean = float(np.mean(target_adjustments))
        self._last_target_adjustment_max = float(np.max(target_adjustments))
        self._last_target_adjustment_count = int(np.sum(target_adjustments > 1e-3))
        
        # DWA执行
        # 前视距离设计：将横向指令转化为动力学友好的前方目标点
        base_lookahead = 3.0  # 基础前视距离（米）
        
        # 检测通道宽度，动态调整DWA参数
        avg_x = np.mean([car.x for car in self.cars])
        detected_very_narrow, passage_width = self._detect_very_narrow_passage(
            avg_x, lookahead=self.very_narrow_lookahead
        )
        if detected_very_narrow:
            self._very_narrow_active = True
            self._very_narrow_seen = True
        if self._very_narrow_active and self._very_narrow_exit_x is not None:
            if avg_x > self._very_narrow_exit_x:
                self._very_narrow_active = False

        is_very_narrow = self._very_narrow_active or detected_very_narrow
        if is_very_narrow and not detected_very_narrow:
            passage_width = self._very_narrow_min_width
        self._last_is_very_narrow = is_very_narrow
        self._last_very_narrow_width = passage_width
        recover_alpha = 1.0
        if (not is_very_narrow) and self._very_narrow_seen and self._very_narrow_exit_x is not None:
            recover_end_x = self._very_narrow_exit_x + self.very_narrow_recover_dist
            recover_alpha = np.clip(
                (avg_x - self._very_narrow_exit_x) / max(self.very_narrow_recover_dist, 1e-3),
                0.0,
                1.0
            )
            if avg_x > recover_end_x:
                recover_alpha = 1.0
        self._very_narrow_recover_alpha = recover_alpha
        
        # 优先级1修复：窄道时降低速度+增大前视，给横向调整更多时间
        upper_bound_global, lower_bound_global = self._get_passage_bounds(
            avg_x, lookahead=self.narrow_lookahead
        )
        passage_width_global = upper_bound_global - lower_bound_global
        formation_width = 1.6  # 原始队形宽度
        is_narrow_passage = passage_width_global < formation_width + 0.5  # 通道<2.1m时触发
        
        if is_narrow_passage and not is_very_narrow:
            # 窄道模式：适中的前视距离，保持队形紧凑
            base_lookahead = 3.5  # 从5.0减小到3.5，保持更紧凑的纵向间距
            base_min_speed = 0.5  # 稍微提高最低速度，减少速度差异
        elif is_very_narrow:
            base_min_speed = 0.3
        else:
            base_min_speed = 0.8

        # ===== 收缩阶段的队形约束（仅窄道模式） =====
        # 注意：安全区域不强制收敛，让RL自由决策避障方向
        if is_narrow_passage and not is_very_narrow:
            # 窄道模式：强制保持完美对称的2x2矩形队形
            # 核心思想：以通道中心为基准，完全对称压缩
            
            # 1. 计算通道中心（障碍物场景通常中心在y=0）
            passage_center = (upper_bound_global + lower_bound_global) / 2
            
            # 2. 队形中心 = 通道中心（不允许RL微调，确保完美对称）
            formation_center = passage_center
            
            # 3. 计算压缩比例，确保队形能通过通道
            scale_factor = self._last_scale_factor
            # 确保scale_factor不会太小，保持基本队形
            scale_factor = max(scale_factor, 0.45)
            
            # 4. 直接应用模板偏移 * 压缩比例 + 队形中心
            # 确保完美对称：同符号的y偏移相等
            for i in range(self.num_cars):
                template_y = self.formation_params.template_offsets[i][1]
                target_positions[i, 1] = formation_center + template_y * scale_factor
        
        # 安全区域的队形恢复（非窄道、非极窄时）
        elif not is_very_narrow:
            # 检查是否经历过极窄通道（需要恢复或保持队形）
            recover_alpha = getattr(self, "_very_narrow_recover_alpha", 1.0)
            if self._very_narrow_seen:
                if recover_alpha < 1.0:
                    # 恢复期：逐步从纵队恢复到标准2x2矩形
                    # recover_alpha: 0(刚退出,保持收拢) -> 1(完全恢复到标准矩形)
                    for i in range(self.num_cars):
                        template_y = self.formation_params.template_offsets[i][1]
                        target_positions[i, 1] = template_y * recover_alpha
                else:
                    # 恢复完成后：使用标准位置 + 小幅RL微调
                    # 确保队形恢复到标准矩形，同时允许RL微调
                    for i in range(self.num_cars):
                        template_y = self.formation_params.template_offsets[i][1]
                        rl_offset = self.delta_accumulator[i] * 0.3  # 允许小幅微调
                        target_positions[i, 1] = template_y + np.clip(rl_offset, -0.2, 0.2)
        # 其他安全区域（未经历极窄通道）：让RL自由决策（用于s4/s5等中心障碍物场景）

        self._enforce_row_spacing(target_positions)
        
        # ===== 窄道模式：预计算同排车辆的协调lookahead =====
        # 确保同排两车保持相同的x位置（紧凑矩形队形）
        row_lookahead = {}  # key: row_key, value: lookahead
        if is_narrow_passage and not is_very_narrow:
            for row in self._row_groups:
                if len(row) < 2:
                    continue
                row_key = self.formation_params.template_offsets[row[0]][0]  # 使用x偏移作为key
                # 计算该排车辆的最小前方距离
                min_front_dist = float('inf')
                for car_idx in row:
                    car = self.cars[car_idx]
                    for j, other in enumerate(self.cars):
                        if j not in row:  # 只看其他排的车
                            if other.x > car.x:
                                dist = other.x - car.x
                                min_front_dist = min(min_front_dist, dist)
                # 根据最小前方距离计算该排的统一lookahead
                if min_front_dist < 1.0:
                    row_lookahead[row_key] = 0.8
                elif min_front_dist < 1.8:
                    row_lookahead[row_key] = 1.5
                elif min_front_dist < 2.5:
                    row_lookahead[row_key] = 2.5
                else:
                    row_lookahead[row_key] = base_lookahead
        
        dwa_fallback_count = 0  # 初始化DWA回退计数
        for i, car in enumerate(self.cars):
            # 计算到前方最近队友的距离（窄道时用于减速避免追尾）
            front_teammate_dist = float('inf')
            if is_narrow_passage or is_very_narrow:
                # 使用虚拟x打破同排同x的并列，避免纵队阶段前后车重叠
                virtual_x = car.x + i * 0.01
                for j, other in enumerate(self.cars):
                    if j == i:
                        continue
                    other_virtual_x = other.x + j * 0.01
                    if other_virtual_x > virtual_x:  # 前方队友
                        dist = other_virtual_x - virtual_x
                        front_teammate_dist = min(front_teammate_dist, dist)

            # 收缩/极窄模式下：根据前车距离动态降低最小速度，避免追尾
            min_speed = base_min_speed
            if is_very_narrow:
                # 极窄模式：更激进的减速以形成纵队
                if front_teammate_dist < self.front_buffer_dist:
                    min_speed = 0.0
                elif front_teammate_dist < self.follow_brake_dist:
                    min_speed = min(min_speed, 0.2)
            elif is_narrow_passage:
                # 窄道模式：温和减速，保持队形紧凑
                if front_teammate_dist < 1.0:
                    min_speed = 0.0
                elif front_teammate_dist < 1.5:
                    min_speed = min(min_speed, 0.3)
                # 超过1.5m不降低最小速度，保持队形紧凑
            self.dwa_controllers[i].params.min_speed = min_speed
            
            # 窄道模式下增强DWA的横向跟随能力
            if is_narrow_passage and not is_very_narrow:
                self.dwa_controllers[i].params.rl_direction_weight = 8.0  # 增强横向跟随
            elif is_very_narrow:
                self.dwa_controllers[i].params.rl_direction_weight = 10.0  # 极窄时更强
            else:
                self.dwa_controllers[i].params.rl_direction_weight = 5.0  # 默认值

            # 计算本车的前视距离
            if is_very_narrow:
                # 极窄通道模式：根据【前方队友距离】动态调整lookahead
                # 而不是固定排名，这样当前车走远后，后车会自动跟上
                
                # 根据前方队友距离调整lookahead
                if front_teammate_dist < 1.5:
                    # 前方有队友且很近(< 1.5m)，几乎停止等待
                    lookahead = 0.3
                elif front_teammate_dist < 3.0:
                    # 前方有队友，适度减速 (1.5-3.0m)
                    lookahead = 1.0
                elif front_teammate_dist < 5.0:
                    # 前方队友距离适中，轻微减速 (3.0-5.0m)
                    lookahead = 2.0
                else:
                    # 前方没有队友或距离足够远，正常前进
                    lookahead = base_lookahead
                
                # 目标y：强制收拢到中心线（通道中心是y=0）
                # RL的delta_y作为微调，但范围缩小
                dwa_target_y = 0.0 + self.delta_accumulator[i] * 0.2
                dwa_target_y = np.clip(dwa_target_y, -0.25, 0.25)  # 极窄通道更强力收拢
            else:
                # 正常模式
                lookahead = base_lookahead
                if is_narrow_passage and not is_very_narrow:
                    # 窄道模式：使用同排协调的lookahead，保持矩形队形
                    row_key = self.formation_params.template_offsets[i][0]
                    if row_key in row_lookahead:
                        lookahead = row_lookahead[row_key]
                    # 如果没有预计算值，使用默认的base_lookahead
                recover_alpha = getattr(self, "_very_narrow_recover_alpha", 1.0)
                if self._very_narrow_seen and recover_alpha < 1.0:
                    # 极窄退出后的恢复期：逐步恢复到2x2队形
                    # target_positions已经包含了恢复逻辑，直接使用
                    # 限制范围确保平滑过渡
                    recover_limit = 0.3 + 0.5 * recover_alpha  # 0.3 -> 0.8
                    dwa_target_y = np.clip(target_positions[i, 1], -recover_limit, recover_limit)
                else:
                    # 正常模式或恢复完成后：直接使用target_positions
                    dwa_target_y = target_positions[i, 1]
                
                # 注意：移除_safe_target_y的强制修正
                # 让RL通过奖励信号自己学习收缩队形，而不是靠后处理
                # 目标点越界惩罚在_compute_reward中实现
            
            # DWA目标点
            dwa_target_x = car.x + lookahead
            
            # 获取附近障碍物
            nearby_obs = self._get_nearby_obstacles(car)
            
            # ========== 协调收缩机制（优先级2修复：收缩模式下放宽队友障碍） ==========
            is_contracting = is_narrow_passage
            
            for j, other in enumerate(self.cars):
                if j != i:
                    dx = other.x - car.x
                    dy = other.y - car.y
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    # 前方队友不阻挡追赶（但过近时仍需避碰）
                    is_front_teammate = dx > 1.0
                    front_too_close = is_contracting and (0.0 < dx < self.front_buffer_dist)
                    
                    # 判断是否同排队友（前左/前右 或 后左/后右）
                    same_row = abs(self.formation_params.template_offsets[i][0] - 
                                   self.formation_params.template_offsets[j][0]) < 0.5
                    
                    if dist < 5.0 and (not is_front_teammate or front_too_close):
                        # 收缩模式下的同排队友：已有目标间距硬约束(0.64m)，DWA不需要再避
                        # 这是关键修复：解决DWA可行解不足的问题
                        if is_contracting and same_row:
                            continue  # 不将同排队友作为DWA障碍
                        
                        # 基础避碰半径
                        base_radius = self.vehicle_params.car_radius + 0.15  # 0.42m
                        
                        if is_contracting:
                            teammate_radius = self.vehicle_params.car_radius + 0.05  # 0.32m，更宽松
                            predicted_y = target_positions[j, 1]
                        else:
                            teammate_radius = base_radius
                            predicted_y = other.y
                        
                        car_obs = Obstacle(
                            x=other.x, y=predicted_y,
                            radius=teammate_radius,
                            obs_type="car"
                        )
                        nearby_obs.append(car_obs)
            
            # DWA控制：目标点在前方，preferred_y引导横向移动
            state = (car.x, car.y, car.theta, car.v)
            goal = (dwa_target_x, dwa_target_y)
            # 窄道/极窄/恢复期模式下，preferred_y强制使用目标位置
            recover_alpha = getattr(self, "_very_narrow_recover_alpha", 1.0)
            if is_very_narrow or recover_alpha < 1.0 or is_narrow_passage:
                # 强制使用目标y，确保车辆快速到达目标位置
                preferred_y = dwa_target_y
            else:
                preferred_y = target_positions[i, 1]
            a, delta = self.dwa_controllers[i].compute(state, goal, nearby_obs, preferred_y)

            debug_info = self.dwa_controllers[i].get_debug_info()
            if debug_info.get("fallback", False):
                dwa_fallback_count += 1
            
            # 更新车辆状态
            self.cars[i] = self.bicycle.step(car, (a, delta), self.env_params.sim_dt)
        
        # 记录轨迹
        for i, car in enumerate(self.cars):
            self.trajectories[i].append((car.x, car.y))

        self._last_dwa_fallback_count = dwa_fallback_count
        
        # 计算奖励（优先级3修复：传入安全修正后的target_positions）
        reward, reward_info = self._compute_reward(target_positions)
        
        # 检查终止条件
        terminated, truncated = self._check_done()
        
        obs = self._get_observation()
        info = self._get_info()
        info.update(reward_info)
        
        return obs, reward, terminated, truncated, info
    
    def _get_current_vector(self) -> np.ndarray:
        """获取当前帧的向量观测（24维）
        包含：累积修正量(4) + 车辆y(4) + 车辆速度(4) + 理想y(4) + 通道信息(8)
        注意：障碍物细节由BEV图像提供，向量只提供通道宽度和相对位置
        
        通道信息(8维)：
        - [16]: 近距离通道宽度(5m内) - 归一化
        - [17]: 远距离通道宽度(15m内) - 归一化，用于提前规划
        - [18]: 通道中心y位置 - 归一化
        - [19]: 是否窄道(两侧障碍物) - 0或1
        - [20:24]: 4车相对通道中心的偏移 - 归一化
        """
        ideal_positions = self._compute_ideal_positions()
        
        vector = np.zeros(self.vector_dim_per_frame, dtype=np.float32)
        vector[0:4] = self.delta_accumulator  # 累积修正量(4维dy)
        for i, car in enumerate(self.cars):
            vector[4 + i] = car.y / self.env_params.road_half_width  # 归一化y
            vector[8 + i] = car.v / self.vehicle_params.v_max  # 归一化速度
            vector[12 + i] = ideal_positions[i, 1] / self.env_params.road_half_width  # 理想y
        
        # 添加通道信息：让RL知道前方可通行区域（障碍物细节由BEV图像提供）
        avg_x = np.mean([car.x for car in self.cars])
        
        # 近距离通道（5m内）- 用于紧急避障
        upper_near, lower_near = self._get_passage_bounds(avg_x, lookahead=5.0)
        width_near = upper_near - lower_near
        center_near = (upper_near + lower_near) / 2
        
        # 远距离通道（15m内）- 用于提前规划
        upper_far, lower_far = self._get_passage_bounds(avg_x, lookahead=15.0)
        width_far = upper_far - lower_far
        center_far = (upper_far + lower_far) / 2
        
        # 通道信息(8维)设计：
        # vector[16]: 近距离通道宽度（归一化）
        # vector[17]: 远距离通道宽度（归一化）
        # vector[18]: 通道中心y（归一化）
        # vector[19]: 是否有两侧障碍物(0或1)
        # vector[20:24]: 4车相对通道中心的偏移
        
        has_both_sides = (upper_near < self.env_params.road_half_width - 0.1 and 
                          lower_near > -self.env_params.road_half_width + 0.1)
        
        vector[16] = np.clip(width_near / 5.0, 0, 1)  # 近距离通道宽度
        vector[17] = np.clip(width_far / 5.0, 0, 1)   # 远距离通道宽度
        vector[18] = center_near / self.env_params.road_half_width  # 通道中心位置
        vector[19] = 1.0 if has_both_sides else 0.0   # 是否窄道（两侧障碍物）
        
        for i, car in enumerate(self.cars):
            # 车辆相对通道中心的偏移（归一化）
            y_offset = (car.y - center_near) / 2.0
            vector[20 + i] = np.clip(y_offset, -1, 1)
        
        return vector
    
    def _get_observation(self) -> Dict:
        """生成观测：图像(84x84x4) + 向量(72维=3帧×24维)"""
        # ========== 图像观测 ==========
        grid = np.zeros((4, self.grid_config.size, self.grid_config.size), dtype=np.float32)
        
        # 前向偏置观测窗口：让图像主要显示前方信息
        # 后方信息无用，前方信息高价值（自行车模型需要提前规划变道）
        center_x = np.mean([car.x for car in self.cars]) + 5.0  # 前移5m
        center_y = np.mean([car.y for car in self.cars])
        
        # 通道0: 障碍物层
        grid[0] = self._render_obstacles(center_x, center_y)
        
        # 通道1: 道路边界层
        grid[1] = self._render_boundaries(center_x, center_y)
        
        # 通道2: 队友位置层
        grid[2] = self._render_cars(center_x, center_y)
        
        # 通道3: 理想队形层
        grid[3] = self._render_ideal_formation(center_x, center_y)
        
        # ========== 向量观测（帧堆叠） ==========
        # 获取当前帧向量
        current_vector = self._get_current_vector()
        
        # 更新帧堆叠缓冲区（deque自动移除最旧帧，O(1)）
        self.vector_history.append(current_vector.copy())
        
        # 堆叠所有帧（最旧在前，最新在后）
        stacked_vector = np.concatenate(list(self.vector_history))
        
        return {"image": grid, "vector": stacked_vector}
    
    def _render_obstacles(self, center_x: float, center_y: float) -> np.ndarray:
        """渲染障碍物层"""
        layer = np.zeros((self.grid_config.size, self.grid_config.size), dtype=np.float32)
        
        for obs in self.obstacles:
            # 转换到栅格坐标
            rel_x = obs.x - center_x
            rel_y = obs.y - center_y
            
            # 检查是否在视野范围内
            if abs(rel_x) > self.grid_config.range_x + 5:
                continue
            if abs(rel_y) > self.grid_config.range_y + 5:
                continue
            
            # 计算栅格范围
            half_w = (obs.width if obs.width > 0 else obs.radius * 2) / 2
            half_h = (obs.height if obs.height > 0 else obs.radius * 2) / 2
            
            # 转换到栅格索引
            x_min = int((rel_x - half_w + self.grid_config.range_x) / self.grid_config.resolution)
            x_max = int((rel_x + half_w + self.grid_config.range_x) / self.grid_config.resolution)
            y_min = int((rel_y - half_h + self.grid_config.range_y) / self.grid_config.resolution)
            y_max = int((rel_y + half_h + self.grid_config.range_y) / self.grid_config.resolution)
            
            # 裁剪到有效范围
            x_min = max(0, x_min)
            x_max = min(self.grid_config.size, x_max)
            y_min = max(0, y_min)
            y_max = min(self.grid_config.size, y_max)
            
            # 填充
            layer[y_min:y_max, x_min:x_max] = 1.0
        
        return layer
    
    def _render_boundaries(self, _cx: float, cy: float) -> np.ndarray:
        """渲染道路边界层（向量化）"""
        layer = np.zeros((self.grid_config.size, self.grid_config.size), dtype=np.float32)
        
        road_hw = self.env_params.road_half_width
        
        # 向量化：计算所有行对应的世界y坐标
        row_indices = np.arange(self.grid_config.size)
        world_ys = cy - self.grid_config.range_y + row_indices * self.grid_config.resolution
        
        # 超出道路边界的行标记为1
        out_of_bounds = (world_ys > road_hw) | (world_ys < -road_hw)
        layer[out_of_bounds, :] = 1.0
        
        return layer
    
    def _render_cars(self, cx: float, cy: float) -> np.ndarray:
        """渲染车辆位置层（向量化）"""
        layer = np.zeros((self.grid_config.size, self.grid_config.size), dtype=np.float32)
        size = self.grid_config.size
        
        for car in self.cars:
            rel_x = car.x - cx
            rel_y = car.y - cy
            
            # 转换到栅格索引
            gx = int((rel_x + self.grid_config.range_x) / self.grid_config.resolution)
            gy = int((rel_y + self.grid_config.range_y) / self.grid_config.resolution)
            
            # 向量化绘制3x3区域
            x_min, x_max = max(0, gx - 1), min(size, gx + 2)
            y_min, y_max = max(0, gy - 1), min(size, gy + 2)
            layer[y_min:y_max, x_min:x_max] = 1.0
        
        return layer
    
    def _render_ideal_formation(self, cx: float, cy: float) -> np.ndarray:
        """渲染理想队形层（向量化）"""
        layer = np.zeros((self.grid_config.size, self.grid_config.size), dtype=np.float32)
        size = self.grid_config.size
        
        ideal_pos = self._compute_ideal_positions()
        
        for pos in ideal_pos:
            rel_x = pos[0] - cx
            rel_y = pos[1] - cy
            
            gx = int((rel_x + self.grid_config.range_x) / self.grid_config.resolution)
            gy = int((rel_y + self.grid_config.range_y) / self.grid_config.resolution)
            
            # 向量化绘制3x3区域
            x_min, x_max = max(0, gx - 1), min(size, gx + 2)
            y_min, y_max = max(0, gy - 1), min(size, gy + 2)
            layer[y_min:y_max, x_min:x_max] = 0.5  # 用较暗的颜色
        
        return layer
    
    def _compute_ideal_positions(self) -> np.ndarray:
        """计算理想队形位置（带缓存）
        
        关键改进：根据前方通道宽度动态调整理想位置
        这样队形惩罚不会惩罚"正确的收缩行为"
        """
        # 缓存检查
        if self._cache_step == self.step_count and self._cached_ideal_positions is not None:
            return self._cached_ideal_positions
        # 以领航点为基准
        positions = np.zeros((self.num_cars, 2))
        
        # 检测前方通道（提前15m规划）
        avg_x = np.mean([car.x for car in self.cars])
        upper_bound, lower_bound = self._get_passage_bounds(avg_x, lookahead=self.narrow_lookahead)
        passage_width = upper_bound - lower_bound
        passage_center = (upper_bound + lower_bound) / 2
        
        # 计算原始队形宽度（template_offsets中y的最大最小差）
        original_y_offsets = [offset[1] for offset in self.formation_params.template_offsets]
        original_width = max(original_y_offsets) - min(original_y_offsets)
        
        # 计算需要的缩放因子
        # 注意：passage_width已经是有效通道（_get_passage_bounds已减去car_radius）
        # 只需要留出安全余量，不要再减car_radius
        safe_passage_width = passage_width - 0.2  # 只减安全余量
        
        # 关键修复：scale_factor最小值必须确保收缩后横向间距 >= 碰撞阈值
        # 碰撞阈值 = 2 * car_radius = 0.54m
        # 安全余量 = 0.1m
        # 最小安全间距 = 0.64m
        # 最小scale_factor = 0.64 / original_width = 0.64 / 1.6 = 0.4
        min_safe_lateral = 2 * self.vehicle_params.car_radius + 0.1  # 0.64m
        min_scale_factor = min_safe_lateral / original_width  # 0.4
        
        if safe_passage_width < original_width:
            # 需要收缩：计算缩放因子，但不能小于安全最小值
            scale_factor = max(min_scale_factor, safe_passage_width / original_width)
        else:
            # 不需要收缩
            scale_factor = 1.0

        # 记录通道与缩放信息（诊断用）
        self._last_passage_width = passage_width
        self._last_passage_center = passage_center
        self._last_scale_factor = scale_factor
        
        for i in range(self.num_cars):
            # 从编队偏移获取相对位置
            offset = self.formation_params.template_offsets[i]
            positions[i, 0] = self.leader_x + offset[0]
            
            # 动态调整y位置：缩放 + 中心偏移
            # 原始y * 缩放因子 + 通道中心偏移
            positions[i, 1] = offset[1] * scale_factor + passage_center
        
        # 更新缓存
        self._cached_ideal_positions = positions
        self._cache_step = self.step_count
        
        return positions
    
    def _get_nearby_obstacles(self, car: VehicleState) -> List[Obstacle]:
        """获取车辆附近的障碍物"""
        nearby = []
        for obs in self.obstacles:
            if -5 < obs.x - car.x < 20:
                nearby.append(obs)
        return nearby
    
    def _safe_target_y(self, car: VehicleState, target_y: float, lookahead: float) -> float:
        """
        安全检查：确保目标y不在障碍物内
        
        只有当检测到前方有"两侧障碍物"（窄道场景）时，才自动将目标点修正到通道内
        单侧障碍物场景让RL自己决定如何避障，避免破坏队形
        """
        # 检查前方lookahead范围内的障碍物
        target_x = car.x + lookahead
        
        upper_bound = self.env_params.road_half_width  # 默认上边界
        lower_bound = -self.env_params.road_half_width  # 默认下边界
        has_upper_obs = False
        has_lower_obs = False
        
        for obs in self.obstacles:
            # 只检查前方障碍物
            if obs.x - obs.width/2 > target_x + 2 or obs.x + obs.width/2 < car.x:
                continue
            
            if obs.width > 0 and obs.height > 0:
                # 矩形障碍物
                half_h = obs.height / 2
                obs_top = obs.y + half_h
                obs_bottom = obs.y - half_h
                
                if obs.y > 0:
                    # 上方障碍物，限制上边界
                    upper_bound = min(upper_bound, obs_bottom - self.vehicle_params.car_radius - 0.1)
                    has_upper_obs = True
                else:
                    # 下方障碍物，限制下边界
                    lower_bound = max(lower_bound, obs_top + self.vehicle_params.car_radius + 0.1)
                    has_lower_obs = True
        
        # 只有两侧都有障碍物时（窄道场景），才强制修正目标点
        # 单侧障碍物让RL自己决定避障方向，保持队形
        if has_upper_obs and has_lower_obs and upper_bound > lower_bound:
            # 窄道场景：强制将目标y限制在通道内
            safe_y = np.clip(target_y, lower_bound, upper_bound)
            return safe_y
        else:
            # 单侧障碍物或无障碍物：保持RL输出不变
            return target_y
    
    def _get_passage_bounds(self, current_x: float, lookahead: float = 10.0) -> Tuple[float, float]:
        """
        获取前方通道的上下边界（带缓存）
        
        处理三种情况：
        1. 上方障碍物：限制upper_bound
        2. 下方障碍物：限制lower_bound
        3. 中间障碍物(y≈0)：不改变边界，让RL决定上下绕行
        
        Returns:
            (upper_bound, lower_bound): 通道的上下边界y值
        """
        # 缓存检查
        cache_key = (round(current_x, 1), lookahead)
        if self._cache_step == self.step_count and cache_key in self._cached_passage_bounds:
            return self._cached_passage_bounds[cache_key]
        upper_bound = self.env_params.road_half_width
        lower_bound = -self.env_params.road_half_width
        
        for obs in self.obstacles:
            # 检查前方lookahead范围内的障碍物
            if obs.x - obs.width/2 > current_x + lookahead or obs.x + obs.width/2 < current_x:
                continue
            
            if obs.width > 0 and obs.height > 0:
                half_h = obs.height / 2
                obs_top = obs.y + half_h
                obs_bottom = obs.y - half_h
                
                # 判断障碍物位置：上方、下方、还是中间
                # 中间障碍物(跨越y=0)需要特殊处理
                is_center_obstacle = obs_bottom < 0.5 and obs_top > -0.5  # 障碍物跨越中心区域
                
                if is_center_obstacle:
                    # 中间障碍物：不改变通道边界
                    # 队形需要分成上下两部分绕行，由RL决定
                    # 保持默认边界，让动态理想位置不收缩
                    pass
                elif obs.y > 0:
                    # 上方障碍物：限制上边界
                    upper_bound = min(upper_bound, obs_bottom - self.vehicle_params.car_radius)
                else:
                    # 下方障碍物：限制下边界
                    lower_bound = max(lower_bound, obs_top + self.vehicle_params.car_radius)
        
        # 更新缓存
        self._cached_passage_bounds[cache_key] = (upper_bound, lower_bound)
        
        return upper_bound, lower_bound
    
    
    def _compute_reward(self, target_positions: np.ndarray) -> Tuple[float, Dict]:
        """
        重新设计的奖励函数
        核心目标：到达终点 + 保持队形 + 安全时恢复队形 + 不碰撞
        优先级3修复：接收安全修正后的target_positions，确保奖励与执行对齐
        """
        reward = 0.0
        info = {}
        
        # 计算理想队形位置
        ideal_positions = self._compute_ideal_positions()
        
        # ========== 1. 碰撞检测与距离场惩罚 ==========
        collision = False
        min_obstacle_dist = float('inf')
        min_boundary_dist = float('inf')
        min_car_dist = float('inf')
        
        for i, car in enumerate(self.cars):
            # 检测障碍物
            for obs in self.obstacles:
                if obs.width > 0 and obs.height > 0:
                    half_w, half_h = obs.width / 2, obs.height / 2
                    dx = max(0, abs(car.x - obs.x) - half_w)
                    dy = max(0, abs(car.y - obs.y) - half_h)
                    dist = np.sqrt(dx**2 + dy**2)
                else:
                    dist = np.sqrt((car.x - obs.x)**2 + (car.y - obs.y)**2) - obs.radius
                
                min_obstacle_dist = min(min_obstacle_dist, dist)
                if dist < self.vehicle_params.car_radius:
                    collision = True
            
            # 检测边界
            boundary_dist = self.env_params.road_half_width - abs(car.y)
            min_boundary_dist = min(min_boundary_dist, boundary_dist)
            if boundary_dist < self.vehicle_params.car_radius:
                collision = True
            
            # 检测车车碰撞
            for j in range(i + 1, self.num_cars):
                other = self.cars[j]
                car_dist = np.sqrt((car.x - other.x)**2 + (car.y - other.y)**2)
                min_car_dist = min(min_car_dist, car_dist)
                if car_dist < self.vehicle_params.car_radius * 2:
                    collision = True
        
        # 记录碰撞类型用于诊断
        collision_type = None
        if min_obstacle_dist < self.vehicle_params.car_radius:
            collision_type = 'obstacle'
            collision = True
        if min_boundary_dist < self.vehicle_params.car_radius:
            collision_type = 'boundary'
            collision = True
        if min_car_dist < self.vehicle_params.car_radius * 2:
            collision_type = 'car_car'
            collision = True
        
        if collision:
            reward -= 500.0  # 碰撞严重惩罚（增大到500）
            info['collision'] = True
            # 记录碰撞位置用于诊断
            info['collision_x'] = np.mean([car.x for car in self.cars])
            info['collision_ys'] = [car.y for car in self.cars]
            info['collision_type'] = collision_type
            info['min_obstacle_dist'] = min_obstacle_dist
            info['min_boundary_dist'] = min_boundary_dist
            info['min_car_dist'] = min_car_dist
            return reward, info
        info['collision'] = False

        info['min_obstacle_dist'] = min_obstacle_dist
        info['min_boundary_dist'] = min_boundary_dist
        info['min_car_dist'] = min_car_dist
        
        # 提前计算avg_x供后续使用
        avg_x = np.mean([car.x for car in self.cars])
        
        # ========== 目标点越界惩罚（让RL学会收缩） ==========
        # 优先级3修复：使用安全修正后的target_positions，确保奖励与执行对齐
        upper_bound, lower_bound = self._get_passage_bounds(avg_x, lookahead=self.narrow_lookahead)
        passage_width = upper_bound - lower_bound
        
        # 只有当通道宽度小于队形宽度时才启用此惩罚（窄道场景）
        formation_width = 1.6  # 2x2队形的横向宽度
        if passage_width < formation_width + 0.5:  # 有一定余量
            target_out_of_bounds_penalty = 0.0
            for i, car in enumerate(self.cars):
                # 使用安全修正后的目标y（与DWA实际跟踪的目标一致）
                target_y = target_positions[i, 1]
                
                # 计算目标点超出通道的程度
                if target_y > upper_bound:
                    overshoot = target_y - upper_bound
                    target_out_of_bounds_penalty += 15.0 * overshoot ** 2
                elif target_y < lower_bound:
                    overshoot = lower_bound - target_y
                    target_out_of_bounds_penalty += 15.0 * overshoot ** 2
            
            reward -= target_out_of_bounds_penalty
            info['target_out_of_bounds_penalty'] = target_out_of_bounds_penalty
        else:
            info['target_out_of_bounds_penalty'] = 0.0
        
        # ========== 2. 距离场惩罚（指数惩罚，解决梯度竞争问题） ==========
        # 使用指数惩罚：距离越近，惩罚梯度越大，确保最后一刻斥力>引力
        detection_range = 5.0  # 检测范围
        
        # 障碍物接近惩罚（指数形式）
        if min_obstacle_dist < detection_range:
            # 指数惩罚：exp(k*(threshold-dist))，距离越近梯度越大
            if min_obstacle_dist < 0.5:
                # 极近距离：强指数惩罚
                proximity_penalty = 3.0 * np.exp(2.0 * (0.5 - min_obstacle_dist))
            else:
                # 一般距离：平滑递增惩罚
                proximity_penalty = 2.0 * (1.0 - min_obstacle_dist / detection_range) ** 2
            reward -= proximity_penalty
            info['obstacle_proximity_penalty'] = proximity_penalty
        else:
            info['obstacle_proximity_penalty'] = 0.0
        
        # 边界接近惩罚（同样使用指数形式）
        boundary_detection = 2.0
        if min_boundary_dist < boundary_detection:
            if min_boundary_dist < 0.3:
                boundary_penalty = 3.0 * np.exp(2.0 * (0.3 - min_boundary_dist))
            else:
                boundary_penalty = 2.0 * (1.0 - min_boundary_dist / boundary_detection) ** 2
            reward -= boundary_penalty
            info['boundary_proximity_penalty'] = boundary_penalty
        else:
            info['boundary_proximity_penalty'] = 0.0
        
        # 车车接近惩罚（关键修复：提供渐进惩罚信号防止车车碰撞）
        # 碰撞阈值 = 2*car_radius = 0.54m
        # 安全距离 = 0.64m（碰撞阈值 + 0.1m余量）
        car_car_safe_dist = 2 * self.vehicle_params.car_radius + 0.1  # 0.64m
        car_car_detection = 1.5  # 检测范围
        if min_car_dist < car_car_detection:
            if min_car_dist < car_car_safe_dist:
                # 危险距离：强指数惩罚
                car_car_penalty = 5.0 * np.exp(3.0 * (car_car_safe_dist - min_car_dist))
            else:
                # 警告距离：平滑惩罚
                car_car_penalty = 2.0 * (1.0 - (min_car_dist - car_car_safe_dist) / (car_car_detection - car_car_safe_dist)) ** 2
            reward -= car_car_penalty
            info['car_car_proximity_penalty'] = car_car_penalty
        else:
            info['car_car_proximity_penalty'] = 0.0
        
        # ========== 3. 前进奖励（主要驱动力） ==========
        # avg_x已在前面计算
        min_x = min(car.x for car in self.cars)  # 关注最慢的车
        
        progress = avg_x - getattr(self, '_last_avg_x', avg_x)
        self._last_avg_x = avg_x
        
        # 前进奖励（增大到12，确保前进动力>避障惩罚）
        progress_reward = 12.0 * progress
        reward += progress_reward
        info['progress_reward'] = progress_reward
        
        # 惩罚掉队（放宽阈值到3.5m，避障时允许更大纵向差距）
        lagging_penalty = 0.05 * max(0, avg_x - min_x - 3.5)  # v4.4权重
        reward -= lagging_penalty
        info['lagging_penalty'] = lagging_penalty
        
        # ========== 4. 队形误差惩罚（形状/尺度分离） ==========
        # 核心思路：允许队形缩放（窄门场景），只惩罚形状畸变（非矩形）
        
        # 获取4车位置
        positions = np.array([[car.x, car.y] for car in self.cars])
        
        # ===== 极窄通道检测：一字长蛇阵模式 =====
        is_very_narrow = getattr(self, "_last_is_very_narrow", False)
        passage_width = getattr(self, "_last_very_narrow_width", float('inf'))
        if not is_very_narrow:
            is_very_narrow, passage_width = self._detect_very_narrow_passage(
                avg_x, lookahead=self.very_narrow_lookahead
            )
        info['is_very_narrow'] = is_very_narrow
        info['passage_width'] = passage_width
        
        if is_very_narrow:
            # ===== 极窄通道模式：使用纵队奖励替代队形惩罚 =====
            column_reward, column_info = self._compute_column_formation_reward(positions)
            reward += column_reward
            info.update(column_info)
            
            # 不使用常规队形惩罚，但仍需定义position_error供终点判断使用
            position_error = 0.0  # 极窄通道模式下不惩罚位置误差
            info['formation_penalty'] = 0.0
            info['shape_error'] = 0.0
            info['scale_error'] = 0.0
            info['position_error'] = 0.0
            info['danger_alpha'] = 1.0  # 完全危险模式
            info['recovery_bonus'] = 0.0
            info['column_mode'] = True
        else:
            # ===== 常规模式：使用原有队形惩罚 =====
            info['column_mode'] = False
            
            # 计算形状误差（基于几何约束）
            shape_error = self._compute_shape_error(positions)
            
            # 计算尺度误差（与理想队形的缩放差异）- 轻微惩罚
            scale_error = self._compute_scale_error(positions, ideal_positions)
            
            # 计算位置误差（y方向偏离）- 用于恢复队形
            position_error = 0.0
            for i, car in enumerate(self.cars):
                y_error = abs(car.y - ideal_positions[i, 1])
                position_error += y_error ** 2
            
            # ========== 平滑危险因子alpha（解决Value Function震荡问题） ==========
            # 用连续alpha替代if/else硬切换，满足神经网络连续性假设
            detection_range_formation = 15.0
            alpha = 1.0 - np.clip(min_obstacle_dist / detection_range_formation, 0.0, 1.0)
            # alpha: 0(安全) ~ 1(危险)
            
            # 动态权重：危险时允许变形，但保持位置跟随
            w_shape = 0.35 * (1.0 - alpha) + 0.2 * alpha  # 安全时更强调形状
            w_scale = 0.2 * (1.0 - alpha) + 0.08 * alpha
            # 关键修复：危险时也要保持position惩罚，让RL跟随动态ideal收缩
            # 原来：w_pos = 0.5*(1-alpha) + 0*alpha，导致alpha=1时w_pos=0
            # 现在：危险时仍保持0.3的权重，确保收缩信号
            w_pos = 1.2 * (1.0 - alpha) + 0.45 * alpha
            
            formation_penalty = w_shape * shape_error + w_scale * scale_error + w_pos * position_error
            recover_alpha = getattr(self, "_very_narrow_recover_alpha", 1.0)
            if recover_alpha < 1.0:
                # 极窄退出后的缓冲期：逐步恢复队形惩罚
                formation_penalty *= (0.3 + 0.7 * recover_alpha)
            
            reward -= formation_penalty
            info['formation_penalty'] = formation_penalty
            info['shape_error'] = shape_error
            info['scale_error'] = scale_error
            info['position_error'] = np.sqrt(position_error)
            info['danger_alpha'] = alpha
            
            # ========== 5. 队形恢复奖励 ==========
            # 当安全(alpha<0.2)且队形接近理想时，给予额外奖励
            if alpha < 0.2 and np.sqrt(position_error) < 0.3:
                recovery_bonus = 3.0
                reward += recovery_bonus
                info['recovery_bonus'] = recovery_bonus
            else:
                info['recovery_bonus'] = 0.0
        
        # ========== 6. 到达终点奖励 ==========
        if all(car.x >= self.env_params.goal_x for car in self.cars):
            # 所有车都到达终点
            reward += 500.0
            info['goal_reached'] = True
            
            # 额外奖励：终点时队形保持得好
            if np.sqrt(position_error) < 0.5:
                reward += 100.0
                info['perfect_formation_bonus'] = True
        else:
            info['goal_reached'] = False
        
        return reward, info
    
    def _check_obstacle_ahead(self, current_x: float, distance: float = 15.0) -> bool:
        """检测前方是否有障碍物"""
        for obs in self.obstacles:
            rel_x = obs.x - current_x
            if 0 < rel_x < distance:
                return True
        return False

    def _build_very_narrow_segments(self) -> None:
        """预计算极窄通道区间，用于跨门保持纵队"""
        self._very_narrow_segments = []
        upper_obs = [
            obs for obs in self.obstacles
            if obs.width > 0 and obs.height > 0 and obs.y > 0
        ]
        lower_obs = [
            obs for obs in self.obstacles
            if obs.width > 0 and obs.height > 0 and obs.y < 0
        ]

        for upper in upper_obs:
            upper_start = upper.x - upper.width / 2
            upper_end = upper.x + upper.width / 2
            for lower in lower_obs:
                lower_start = lower.x - lower.width / 2
                lower_end = lower.x + lower.width / 2

                overlap = min(upper_end, lower_end) - max(upper_start, lower_start)
                if overlap < -0.5:
                    continue

                passage_width = (upper.y - upper.height / 2) - (lower.y + lower.height / 2)
                if passage_width < 1.5:
                    start_x = min(upper_start, lower_start)
                    end_x = max(upper_end, lower_end)
                    self._very_narrow_segments.append((start_x, end_x, passage_width))

        if self._very_narrow_segments:
            self._very_narrow_exit_x = max(end for _, end, _ in self._very_narrow_segments)
            self._very_narrow_exit_x += self.very_narrow_exit_margin
            self._very_narrow_min_width = min(width for _, _, width in self._very_narrow_segments)
        else:
            self._very_narrow_exit_x = None
            self._very_narrow_min_width = float('inf')

    def _build_row_groups(self) -> List[List[int]]:
        """根据编队模板构建同排分组"""
        rows: Dict[float, List[int]] = {}
        for i, offset in enumerate(self.formation_params.template_offsets):
            row_key = round(offset[0], 1)
            rows.setdefault(row_key, []).append(i)
        return list(rows.values())

    def _enforce_row_spacing(self, target_positions: np.ndarray) -> None:
        """确保同排车辆横向间距不小于安全距离"""
        min_safe_dist = 2 * self.vehicle_params.car_radius + 0.1  # 0.64m
        for row in self._row_groups:
            if len(row) < 2:
                continue
            # 只处理2车同排的情况
            i, j = row[0], row[1]
            y_dist = target_positions[i, 1] - target_positions[j, 1]
            if abs(y_dist) < min_safe_dist:
                center_y = (target_positions[i, 1] + target_positions[j, 1]) / 2
                if y_dist >= 0:
                    target_positions[i, 1] = center_y + min_safe_dist / 2
                    target_positions[j, 1] = center_y - min_safe_dist / 2
                else:
                    target_positions[i, 1] = center_y - min_safe_dist / 2
                    target_positions[j, 1] = center_y + min_safe_dist / 2
    
    def _detect_very_narrow_passage(self, current_x: float, lookahead: float = 10.0) -> Tuple[bool, float]:
        """
        检测是否在极窄通道区域内（需要一字长蛇阵的场景）
        
        改进：使用预计算的极窄区间，在多个障碍物之间也保持极窄模式
        
        Returns:
            (is_very_narrow, passage_width)
            - is_very_narrow: 是否在极窄通道区域内
            - passage_width: 通道宽度
        """
        # 方法1：检查是否在预计算的极窄区间内（包括提前量和滞后量）
        pre_margin = 5.0   # 提前进入极窄模式的距离
        post_margin = 3.0  # 离开障碍物后保持极窄模式的距离
        
        for start_x, end_x, width in self._very_narrow_segments:
            if current_x >= start_x - pre_margin and current_x <= end_x + post_margin:
                return True, width
        
        # 方法2：如果有多个极窄区间，检查是否在它们之间
        # 例如 s6_very_narrow 有两个障碍物组，在它们之间也应保持极窄模式
        if len(self._very_narrow_segments) >= 2:
            first_end = min(end for _, end, _ in self._very_narrow_segments)
            last_start = max(start for start, _, _ in self._very_narrow_segments)
            if current_x >= first_end - post_margin and current_x <= last_start + pre_margin:
                # 在两个极窄区间之间，保持极窄模式
                return True, self._very_narrow_min_width
        
        # 方法3：原有的前方检测逻辑（作为后备）
        upper_obs = None
        lower_obs = None
        
        for obs in self.obstacles:
            rel_x = obs.x - current_x
            if -5 < rel_x < lookahead and obs.width > 0:
                if obs.y > 0:
                    upper_obs = obs
                else:
                    lower_obs = obs
        
        if upper_obs is not None and lower_obs is not None:
            upper_bottom = upper_obs.y - upper_obs.height / 2
            lower_top = lower_obs.y + lower_obs.height / 2
            passage_width = upper_bottom - lower_top
            
            if passage_width < 1.5:
                return True, passage_width
        
        return False, float('inf')
    
    def _compute_column_formation_reward(self, positions: np.ndarray) -> Tuple[float, Dict]:
        """
        计算纵队（一字长蛇阵）奖励
        
        理想纵队：4车y坐标相同（横向对齐），x坐标有间距（纵向排列）
        """
        info = {}
        reward = 0.0
        
        # 获取所有车的y坐标和x坐标
        ys = positions[:, 1]
        xs = positions[:, 0]
        
        # 1. 横向聚拢奖励：y坐标标准差越小越好
        y_std = np.std(ys)
        y_range = np.max(ys) - np.min(ys)
        
        # 理想情况：所有车的y坐标差距 < 0.5m
        if y_range < 0.5:
            # 完美纵队，给予奖励
            column_reward = 3.0 * (1.0 - y_range / 0.5)
            reward += column_reward
            info['column_alignment_reward'] = column_reward
        elif y_range < 1.5:
            # 正在收缩，小奖励
            column_reward = 1.0 * (1.5 - y_range) / 1.5
            reward += column_reward
            info['column_alignment_reward'] = column_reward
        else:
            # 还没开始收缩，轻微惩罚引导
            info['column_alignment_reward'] = 0.0
        
        # 2. 纵向间距保持：x方向应该有合理间距（避免追尾）
        x_sorted = np.sort(xs)
        x_gaps = np.diff(x_sorted)
        
        # 理想间距：1.0m ~ 3.0m
        gap_penalty = 0.0
        for gap in x_gaps:
            if gap < 0.5:  # 太近，可能追尾
                gap_penalty += 1.0 * (0.5 - gap)
            elif gap > 5.0:  # 太远，队形分散
                gap_penalty += 0.3 * (gap - 5.0)
        
        reward -= gap_penalty
        info['column_gap_penalty'] = gap_penalty
        
        # 3. 中心线位置奖励：鼓励纵队在道路中心附近
        y_mean = np.mean(ys)
        center_penalty = 0.2 * abs(y_mean)  # 越靠近中心越好
        reward -= center_penalty
        info['column_center_penalty'] = center_penalty
        
        info['y_range'] = y_range
        info['y_std'] = y_std
        
        return reward, info
    
    def _compute_shape_error(self, positions: np.ndarray) -> float:
        """
        计算形状误差（基于几何约束）
        
        矩形的几何性质：
        1. 对边平行：车0-车1 平行于 车2-车3
        2. 邻边垂直：车0-车1 垂直于 车0-车2
        
        只惩罚"歪扭"（形状畸变），不惩罚"缩放"（尺度变化）
        """
        # 车辆编号: 0=前左, 1=前右, 2=后左, 3=后右
        p0, p1, p2, p3 = positions[0], positions[1], positions[2], positions[3]
        
        # 向量
        v01 = p1 - p0  # 前排左→右
        v23 = p3 - p2  # 后排左→右
        v02 = p2 - p0  # 左侧前→后
        v13 = p3 - p1  # 右侧前→后
        
        # 1. 对边平行误差：v01 和 v23 应该平行
        # 使用叉积衡量平行度（叉积=0表示完全平行）
        cross_front_back = abs(v01[0] * v23[1] - v01[1] * v23[0])
        
        # 归一化（除以向量长度的乘积，避免尺度影响）
        len01 = np.linalg.norm(v01) + 1e-6
        len23 = np.linalg.norm(v23) + 1e-6
        parallel_error_1 = cross_front_back / (len01 * len23)
        
        # 2. 左右两侧也应该平行
        cross_left_right = abs(v02[0] * v13[1] - v02[1] * v13[0])
        len02 = np.linalg.norm(v02) + 1e-6
        len13 = np.linalg.norm(v13) + 1e-6
        parallel_error_2 = cross_left_right / (len02 * len13)
        
        # 3. 邻边垂直误差：v01 和 v02 应该垂直
        # 使用点积衡量垂直度（点积=0表示完全垂直）
        dot_product = abs(v01[0] * v02[0] + v01[1] * v02[1])
        perpendicular_error = dot_product / (len01 * len02)
        
        # 综合形状误差
        shape_error = parallel_error_1 + parallel_error_2 + perpendicular_error
        
        return shape_error
    
    def _compute_scale_error(self, positions: np.ndarray, ideal_positions: np.ndarray) -> float:
        """
        计算尺度误差（与理想队形的缩放差异）
        
        只衡量整体缩放程度，不关心具体位置
        """
        # 计算当前队形的横向/纵向间距
        p0, p1, p2, _p3 = positions[0], positions[1], positions[2], positions[3]
        current_lateral = np.linalg.norm(p1 - p0)  # 横向间距
        current_longitudinal = np.linalg.norm(p2 - p0)  # 纵向间距
        
        # 计算理想队形的横向/纵向间距
        ip0, ip1, ip2 = ideal_positions[0], ideal_positions[1], ideal_positions[2]
        ideal_lateral = np.linalg.norm(ip1 - ip0)
        ideal_longitudinal = np.linalg.norm(ip2 - ip0)
        
        # 计算缩放比例偏差
        if ideal_lateral > 0 and ideal_longitudinal > 0:
            lateral_ratio = current_lateral / ideal_lateral
            longitudinal_ratio = current_longitudinal / ideal_longitudinal
            
            # 尺度误差：偏离1.0的程度
            scale_error = (abs(lateral_ratio - 1.0) + abs(longitudinal_ratio - 1.0))
        else:
            scale_error = 0.0
        
        return scale_error
    
    def _check_done(self) -> Tuple[bool, bool]:
        """检查是否终止"""
        terminated = False
        truncated = False
        
        # 碰撞终止
        for i, car in enumerate(self.cars):
            for obs in self.obstacles:
                if obs.width > 0 and obs.height > 0:
                    # 矩形障碍物：计算到矩形边缘的距离
                    half_w = obs.width / 2
                    half_h = obs.height / 2
                    dx = max(0, abs(car.x - obs.x) - half_w)
                    dy = max(0, abs(car.y - obs.y) - half_h)
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < self.vehicle_params.car_radius:
                        terminated = True
                        return terminated, truncated
                else:
                    # 圆形障碍物
                    dist = np.sqrt((car.x - obs.x)**2 + (car.y - obs.y)**2)
                    if dist < obs.radius + self.vehicle_params.car_radius:
                        terminated = True
                        return terminated, truncated
            
            if abs(car.y) > self.env_params.road_half_width - self.vehicle_params.car_radius:
                terminated = True
                return terminated, truncated
            
            for j in range(i + 1, self.num_cars):
                other = self.cars[j]
                dist = np.sqrt((car.x - other.x)**2 + (car.y - other.y)**2)
                if dist < self.vehicle_params.car_radius * 2:
                    terminated = True
                    return terminated, truncated
        
        # 到达终点
        if all(car.x >= self.env_params.goal_x for car in self.cars):
            terminated = True
            return terminated, truncated
        
        # 超时
        if self.step_count >= self.max_steps:
            truncated = True
        
        return terminated, truncated
    
    def _get_info(self) -> Dict:
        """获取环境信息"""
        return {
            "step": self.step_count,
            "leader_x": self.leader_x,
            "avg_car_x": np.mean([car.x for car in self.cars]),
            "rl_active": self.rl_active,
            "passage_width": self._last_passage_width,
            "passage_center": self._last_passage_center,
            "scale_factor": self._last_scale_factor,
            "target_adjustment_mean": self._last_target_adjustment_mean,
            "target_adjustment_max": self._last_target_adjustment_max,
            "target_adjustment_count": self._last_target_adjustment_count,
            "dwa_fallback_count": self._last_dwa_fallback_count,
        }
    
    def render(self):
        """渲染（可选实现）"""
        if self.render_mode == "human":
            pass  # TODO: 实现可视化
        return None
    
    def close(self):
        """关闭环境"""
        pass


# 测试
if __name__ == "__main__":
    env = FormationRLEnv(scenario="main", num_cars=4)
    
    obs, info = env.reset()
    print(f"Observation image shape: {obs['image'].shape}")
    print(f"Observation vector shape: {obs['vector'].shape}")
    print(f"Action space: {env.action_space}")
    
    # 测试几步
    for i in range(10):
        action = env.action_space.sample() * 0.1  # 小动作
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.2f}, avg_x={info['avg_car_x']:.1f}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("Environment test passed!")
