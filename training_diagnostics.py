"""
训练诊断脚本

功能：
1. 评估模型在指定场景的成功率/碰撞率/超时率
2. 统计碰撞类型占比（obstacle / boundary / car_car）
3. 按障碍段统计失败原因（在哪一段最容易失败）
4. 导出文本摘要与JSON明细，便于对比不同版本环境与模型

示例：
python training_diagnostics.py --model outputs/curriculum_xxx/stage6_comprehensive_all/best_model.zip --scenario comprehensive_all --episodes 100
python training_diagnostics.py --model models/pretrain_bc.zip --scenario s3_narrow --episodes 50 --stochastic
"""

from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
from stable_baselines3 import PPO

from formation_rl_env import FormationRLEnv
from config import OBSTACLE_SCENARIOS, Obstacle


@dataclass
class Segment:
    """障碍段定义"""
    idx: int
    start_x: float
    end_x: float
    center_x: float
    label: str


def obstacle_x_range(obs: Obstacle) -> Tuple[float, float]:
    """返回障碍物在x方向的覆盖区间"""
    if obs.width and obs.width > 0:
        return obs.x - obs.width / 2.0, obs.x + obs.width / 2.0
    return obs.x - obs.radius, obs.x + obs.radius


def build_segments(scenario: str, merge_gap: float = 6.0) -> List[Segment]:
    """
    将场景障碍物按x聚类成“障碍段”
    - 同一障碍段通常对应一个门/一个避障事件
    - merge_gap 控制相邻障碍组是否合并
    """
    obstacles = OBSTACLE_SCENARIOS.get(scenario, [])
    if not obstacles:
        return []

    ranges = []
    for obs in obstacles:
        s, e = obstacle_x_range(obs)
        ranges.append((s, e, obs))

    ranges.sort(key=lambda x: x[0])

    clusters: List[List[Tuple[float, float, Obstacle]]] = []
    current = [ranges[0]]

    for item in ranges[1:]:
        s, e, _ = item
        _, cur_end, _ = max(current, key=lambda x: x[1])
        if s <= cur_end + merge_gap:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    clusters.append(current)

    segments: List[Segment] = []
    for i, cluster in enumerate(clusters, start=1):
        start_x = min(it[0] for it in cluster)
        end_x = max(it[1] for it in cluster)
        center_x = 0.5 * (start_x + end_x)
        label = f"seg{i}@x={center_x:.1f}"
        segments.append(Segment(i, start_x, end_x, center_x, label))

    return segments


def locate_segment_by_x(x: float, segments: List[Segment], margin: float = 3.0) -> str:
    """根据x定位故障发生在哪个障碍段附近"""
    if not segments:
        return "no_obstacle"

    for seg in segments:
        if seg.start_x - margin <= x <= seg.end_x + margin:
            return seg.label

    dists = [abs(x - seg.center_x) for seg in segments]
    return segments[int(np.argmin(dists))].label


def determine_timeout_segment(max_progress_x: float, segments: List[Segment]) -> str:
    """超时时，判断卡在了哪一段（下一段未通过）"""
    if not segments:
        return "no_obstacle"

    for seg in segments:
        if max_progress_x < seg.end_x + 2.0:
            return seg.label

    return "after_last_segment"


def run_diagnostics(
    model_path: str,
    scenario: str,
    episodes: int,
    deterministic: bool,
    max_steps: int,
    seed: int,
) -> Dict[str, Any]:
    """执行诊断评估并返回统计结果"""
    model = PPO.load(model_path)
    env = FormationRLEnv(scenario=scenario, num_cars=4, max_steps=max_steps, use_rl=True)

    segments = build_segments(scenario)

    outcome_counter = Counter()
    collision_type_counter = Counter()
    failure_segment_counter = Counter()

    episode_details: List[Dict[str, Any]] = []

    rewards = []
    lengths = []
    final_avg_xs = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)

        done = False
        ep_reward = 0.0
        ep_steps = 0

        max_progress_x = -1e9
        collision_x = None
        collision_type = None

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += float(reward)
            ep_steps += 1

            avg_x = float(info.get("avg_car_x", np.mean([car.x for car in env.cars])))
            max_progress_x = max(max_progress_x, avg_x)

            if info.get("collision", False):
                collision_x = float(info.get("collision_x", avg_x))
                collision_type = str(info.get("collision_type", "unknown"))

        # 终局分类
        goal_reached = bool(info.get("goal_reached", False)) or all(
            car.x >= env.env_params.goal_x for car in env.cars
        )

        if goal_reached:
            outcome = "success"
            failure_segment = None
        elif info.get("collision", False):
            outcome = "collision"
            failure_segment = locate_segment_by_x(collision_x if collision_x is not None else max_progress_x, segments)
            collision_type_counter[collision_type or "unknown"] += 1
        else:
            outcome = "timeout"
            failure_segment = determine_timeout_segment(max_progress_x, segments)

        outcome_counter[outcome] += 1
        if failure_segment is not None:
            failure_segment_counter[failure_segment] += 1

        ep_detail = {
            "episode": ep,
            "outcome": outcome,
            "reward": ep_reward,
            "steps": ep_steps,
            "final_avg_x": float(np.mean([car.x for car in env.cars])),
            "max_progress_x": max_progress_x,
            "collision_type": collision_type,
            "collision_x": collision_x,
            "failure_segment": failure_segment,
        }
        episode_details.append(ep_detail)

        rewards.append(ep_reward)
        lengths.append(ep_steps)
        final_avg_xs.append(ep_detail["final_avg_x"])

    env.close()

    total = max(1, episodes)
    summary = {
        "model": model_path,
        "scenario": scenario,
        "episodes": episodes,
        "deterministic": deterministic,
        "max_steps": max_steps,
        "success_rate": outcome_counter["success"] / total,
        "collision_rate": outcome_counter["collision"] / total,
        "timeout_rate": outcome_counter["timeout"] / total,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_final_avg_x": float(np.mean(final_avg_xs)) if final_avg_xs else 0.0,
        "outcome_counts": dict(outcome_counter),
        "collision_type_counts": dict(collision_type_counter),
        "failure_segment_counts": dict(failure_segment_counter),
        "segments": [
            {
                "idx": s.idx,
                "label": s.label,
                "start_x": s.start_x,
                "end_x": s.end_x,
                "center_x": s.center_x,
            }
            for s in segments
        ],
    }

    return {
        "summary": summary,
        "episode_details": episode_details,
    }


def render_text_report(result: Dict[str, Any]) -> str:
    """生成可读文本报告"""
    s = result["summary"]

    lines = []
    lines.append("=" * 72)
    lines.append("UGV 训练诊断报告")
    lines.append("=" * 72)
    lines.append(f"Model      : {s['model']}")
    lines.append(f"Scenario   : {s['scenario']}")
    lines.append(f"Episodes   : {s['episodes']}")
    lines.append(f"Deterministic: {s['deterministic']}")
    lines.append("-" * 72)
    lines.append(
        f"成功率={s['success_rate']*100:.1f}% | 碰撞率={s['collision_rate']*100:.1f}% | 超时率={s['timeout_rate']*100:.1f}%"
    )
    lines.append(
        f"平均回报={s['mean_reward']:.2f} ± {s['std_reward']:.2f} | 平均步数={s['mean_length']:.1f} | 平均最终x={s['mean_final_avg_x']:.2f}"
    )

    lines.append("-" * 72)
    lines.append("碰撞类型占比（仅碰撞回合）:")
    coll = s.get("collision_type_counts", {})
    coll_total = sum(coll.values())
    if coll_total == 0:
        lines.append("  无碰撞")
    else:
        for k, v in sorted(coll.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"  {k:>10}: {v:4d} ({100.0*v/coll_total:.1f}%)")

    lines.append("-" * 72)
    lines.append("按障碍段失败统计（collision + timeout）:")
    seg = s.get("failure_segment_counts", {})
    if not seg:
        lines.append("  无失败回合")
    else:
        for k, v in sorted(seg.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"  {k:>14}: {v:4d}")

    if s.get("segments"):
        lines.append("-" * 72)
        lines.append("障碍段定义:")
        for item in s["segments"]:
            lines.append(
                f"  {item['label']}: x in [{item['start_x']:.1f}, {item['end_x']:.1f}]"
            )

    lines.append("=" * 72)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="UGV训练诊断脚本")
    parser.add_argument("--model", type=str, required=True, help="模型路径（.zip）")
    parser.add_argument("--scenario", type=str, default="comprehensive_all", help="评估场景")
    parser.add_argument("--episodes", type=int, default=100, help="评估回合数")
    parser.add_argument("--max_steps", type=int, default=2000, help="单回合最大步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--stochastic", action="store_true", help="使用随机策略采样（默认deterministic）")
    parser.add_argument("--out_dir", type=str, default="test_results", help="输出目录")
    parser.add_argument("--tag", type=str, default="", help="输出文件标签")

    args = parser.parse_args()

    model_path = args.model
    if not model_path.endswith(".zip") and os.path.exists(model_path + ".zip"):
        model_path = model_path + ".zip"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    result = run_diagnostics(
        model_path=model_path,
        scenario=args.scenario,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    report_text = render_text_report(result)
    print(report_text)

    tag = f"_{args.tag}" if args.tag else ""
    base_name = f"diag_{args.scenario}_ep{args.episodes}{tag}"

    txt_path = os.path.join(args.out_dir, base_name + ".txt")
    json_path = os.path.join(args.out_dir, base_name + ".json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n已输出:\n- {txt_path}\n- {json_path}")


if __name__ == "__main__":
    main()
