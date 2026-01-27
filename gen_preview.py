import os
import sys
import importlib

# 重新加载config模块以获取最新定义
import config
importlib.reload(config)

from config import OBSTACLE_SCENARIOS
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config import EnvParams

scenario = 's1_s2_s5_mixed'
save_path = 'png/preview_s1_s2_s5_mixed.png'

env_params = EnvParams()
obstacles = OBSTACLE_SCENARIOS.get(scenario, [])

# 计算场景范围
max_x = max(obs.x for obs in obstacles) + 20 if obstacles else 100

# 创建图形
fig, ax = plt.subplots(figsize=(16, 4))

# 绘制道路
road_half_width = env_params.road_half_width
ax.fill_between([0, max_x], [-road_half_width, -road_half_width], 
                [road_half_width, road_half_width], 
                color='lightgray', alpha=0.5)

# 道路边界
ax.axhline(y=road_half_width, color='black', linewidth=2)
ax.axhline(y=-road_half_width, color='black', linewidth=2)
ax.axhline(y=0, color='yellow', linewidth=1, linestyle='--', alpha=0.5)

# 绘制障碍物
colors = {'rect': 'red', 'circle': 'orange', 'car': 'blue'}
for obs in obstacles:
    color = colors.get(obs.obs_type, 'red')
    if obs.obs_type in ['rect', 'car']:
        width = obs.width if obs.width > 0 else obs.radius * 2
        height = obs.height if obs.height > 0 else obs.radius * 2
        rect = patches.Rectangle((obs.x - width/2, obs.y - height/2), width, height,
                                  facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
    else:
        circle = patches.Circle((obs.x, obs.y), obs.radius, facecolor=color, 
                                 edgecolor='black', alpha=0.7)
        ax.add_patch(circle)

ax.set_xlim(-5, max_x)
ax.set_ylim(-road_half_width - 1, road_half_width + 1)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'Scenario: {scenario}')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

os.makedirs('png', exist_ok=True)
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()

# 写入结果文件
with open('gen_result.txt', 'w') as f:
    f.write(f'Generated: {save_path}\n')
    f.write(f'File exists: {os.path.exists(save_path)}\n')
