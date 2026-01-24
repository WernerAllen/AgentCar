"""
场景可视化脚本 - 显示道路布局和障碍物
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from config import OBSTACLE_SCENARIOS, CAR_INIT_CONFIGS, EnvParams

def visualize_scenario(scenario: str = "main", save_path: str = None):
    """可视化场景布局"""
    
    env_params = EnvParams()
    obstacles = OBSTACLE_SCENARIOS.get(scenario, [])
    
    # 计算场景范围
    if obstacles:
        max_x = max(obs.x for obs in obstacles) + 20
    else:
        max_x = 100
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # 绘制道路
    road_half_width = env_params.road_half_width
    ax.fill_between([0, max_x], [-road_half_width, -road_half_width], 
                    [road_half_width, road_half_width], 
                    color='lightgray', alpha=0.5, label='Road')
    
    # 绘制道路边界
    ax.axhline(y=road_half_width, color='black', linewidth=2)
    ax.axhline(y=-road_half_width, color='black', linewidth=2)
    
    # 绘制中心线
    ax.axhline(y=0, color='yellow', linewidth=1, linestyle='--', alpha=0.5)
    
    # 绘制障碍物
    colors = {'rect': 'red', 'circle': 'orange', 'car': 'blue', 'cone': 'yellow', 'person': 'green'}
    
    for obs in obstacles:
        color = colors.get(obs.obs_type, 'red')
        
        if obs.obs_type == 'rect' or obs.obs_type == 'car':
            # 矩形障碍物
            width = obs.width if obs.width > 0 else obs.radius * 2
            height = obs.height if obs.height > 0 else obs.radius * 2
            rect = patches.Rectangle(
                (obs.x - width/2, obs.y - height/2),
                width, height,
                linewidth=1, edgecolor='black', facecolor=color, alpha=0.7
            )
            ax.add_patch(rect)
        else:
            # 圆形障碍物
            circle = patches.Circle(
                (obs.x, obs.y), obs.radius,
                linewidth=1, edgecolor='black', facecolor=color, alpha=0.7
            )
            ax.add_patch(circle)
        
        # 标注位置
        ax.annotate(f'{obs.x}m', (obs.x, obs.y), fontsize=8, ha='center', va='bottom')
    
    # 绘制车辆初始位置
    for car_config in CAR_INIT_CONFIGS:
        x, y = car_config['init_pos']
        car = patches.Circle((x, y), 0.3, color=car_config['color'], alpha=0.8)
        ax.add_patch(car)
        ax.annotate(car_config['name'], (x, y+0.5), fontsize=7, ha='center')
    
    # 绘制场景标注
    scene_labels = [
        (0, "Start"),
        (25, "S1:Right Block"),
        (50, "S2:Left Block"),
        (75, "S3:Both Sides"),
        (100, "S4:Small Center"),
        (125, "S5:Large Center"),
    ]
    
    for x, label in scene_labels:
        ax.axvline(x=x, color='gray', linewidth=0.5, linestyle=':')
        ax.text(x, road_half_width + 0.3, label, fontsize=8, ha='center', rotation=0)
    
    # 设置坐标轴
    ax.set_xlim(-5, max_x + 5)
    ax.set_ylim(-road_half_width - 1, road_half_width + 1.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Scene Layout: {scenario}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    legend_elements = [
        patches.Patch(facecolor='red', edgecolor='black', label='Rect/Barrier'),
        patches.Patch(facecolor='orange', edgecolor='black', label='Circle'),
        patches.Patch(facecolor='blue', edgecolor='black', label='Car'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[已保存] 场景图: {save_path}")
    else:
        save_path = f"scenario_{scenario}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[已保存] 场景图: {save_path}")
    
    plt.close()
    return save_path


if __name__ == "__main__":
    print("=" * 50)
    print("场景可视化")
    print("=" * 50)
    
    # 可视化main场景
    path = visualize_scenario("main")
    print(f"\n场景图已生成: {path}")
