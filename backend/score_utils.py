import math
from typing import Dict, List, Sequence, Tuple

# ------------------
# 简易姿势评分工具
# ------------------

# 将关键点划分为不同身体部位
PARTS = {
    "left_arm": ["left_shoulder", "left_elbow", "left_wrist"],
    "right_arm": ["right_shoulder", "right_elbow", "right_wrist"],
    "left_leg": ["left_hip", "left_knee", "left_ankle"],
    "right_leg": ["right_hip", "right_knee", "right_ankle"],
    "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "head": ["nose"],
}

# 每个部位对应的建议语
SUGGESTIONS = {
    "left_arm": "左臂姿势需调整",
    "right_arm": "右臂姿势需调整",
    "left_leg": "左腿姿势需调整",
    "right_leg": "右腿姿势需调整",
    "torso": "躯干不够挺直",
    "head": "注意头部位置",
}

# 当关键点距离超过该阈值时视为满分扣完
DIST_THRESHOLD = 0.2


def _calc_point_score(dist: float) -> float:
    """根据关键点距离计算得分"""
    score = 100.0 - 100.0 * dist / DIST_THRESHOLD
    return max(0.0, min(100.0, score))


def _score_part(
    user_kps: Dict[str, Sequence[float]],
    std_kps: Dict[str, Sequence[float]],
    names: Sequence[str],
) -> float:
    """计算指定部位的平均得分"""
    scores = []
    for name in names:
        u = user_kps.get(name)
        s = std_kps.get(name)
        if u is None or s is None:
            continue
        dist = math.dist(u[:2], s[:2])
        scores.append(_calc_point_score(dist))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def score_pose(
    user_kps: Dict[str, Sequence[float]],
    std_kps: Dict[str, Sequence[float]],
) -> Dict[str, object]:
    """按照部位计算姿势得分并给出建议"""
    per_part_score: Dict[str, int] = {}
    suggestions: List[str] = []
    all_scores: List[float] = []

    for part, names in PARTS.items():
        part_score = _score_part(user_kps, std_kps, names)
        if part_score:
            per_part_score[part] = int(round(part_score))
            all_scores.append(part_score)
            if part_score < 80 and part in SUGGESTIONS:
                suggestions.append(SUGGESTIONS[part])

    overall = int(round(sum(all_scores) / len(all_scores))) if all_scores else 0

    return {
        "score": overall,
        "per_part_score": per_part_score,
        "suggestion": suggestions,
    }
