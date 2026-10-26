"""闭环控制核心算法：确定性 SM-2 记忆衰减重放调度、EMA 节点稳健度更新与客观任务确定性裁决。"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass


@dataclass
class SM2State:
    ef: float = 2.5
    interval_days: float = 0.0
    repetitions: int = 0
    next_review_at: float = 0.0


def score_to_quality(
    is_correct: bool,
    score: float,
    hints_used: int = 0,
    elapsed_sec: float = 0.0,
) -> int:
    """将评估得分 (0~1)、前置辅助阶数与执行耗时映射为 SM-2 衰减质量分 q ∈ [0, 5]。"""
    if not is_correct or score < 0.6:
        if score >= 0.4:
            return 2
        if score >= 0.2:
            return 1
        return 0

    # 裁决通过的情况 (q ∈ {3, 4, 5})
    if hints_used >= 2 or score < 0.75:
        return 3
    if hints_used == 1 or score < 0.9 or elapsed_sec > 120:
        return 4
    return 5


def compute_sm2(
    ef: float,
    interval_days: float,
    repetitions: int,
    quality: int,
    now: float | None = None,
) -> SM2State:
    """标准 SM-2 记忆衰减调度算法（确定性代码计算下一次重放时间）。"""
    ts = now if now is not None else time.time()
    q = max(0, min(5, int(quality)))

    new_ef = ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    new_ef = max(1.3, round(new_ef, 4))

    if q < 3:
        new_reps = 0
        new_interval = 1.0
    else:
        new_reps = repetitions + 1
        if new_reps == 1:
            new_interval = 1.0
        elif new_reps == 2:
            new_interval = 6.0
        else:
            new_interval = round(max(1.0, interval_days * new_ef), 2)

    next_review = ts + new_interval * 86400.0
    return SM2State(
        ef=new_ef,
        interval_days=new_interval,
        repetitions=new_reps,
        next_review_at=next_review,
    )


def update_mastery_ema(
    prev_mastery: float,
    score: float,
    difficulty: float = 0.5,
    alpha: float = 0.3,
) -> float:
    """指数滑动平均（EMA）更新知识节点稳健度指标 m_t ∈ [0, 1]，任务复杂度越高更新权重越大。"""
    s = max(0.0, min(1.0, float(score)))
    d = max(0.1, min(1.0, float(difficulty)))
    # 复杂度权重 w_d ∈ [0.7, 1.3]
    w_d = 0.6 + 0.7 * d
    eff_alpha = max(0.05, min(0.6, alpha * w_d))
    new_m = (1.0 - eff_alpha) * prev_mastery + eff_alpha * s
    return round(max(0.0, min(1.0, new_m)), 4)


def _normalize_choice(text: str) -> str:
    """提取客观单选/多选任务选项标识（如 '我选B'、'选A'、'A. xxx' -> 'A'）。"""
    s = text.strip().upper()
    m = re.match(r"^(?:我?选\s*|我的?(?:答案|回答|结果)(?:是|为)?\s*|答案(?:是|为)?\s*)?([A-F])(?:\b|[.、:：\s]|$)", s)
    if m:
        return m.group(1)
    if len(s) == 1 and s in "ABCDEF":
        return s
    return s


def grade_objective(q_type: str, expected_answer: str, user_answer: str) -> tuple[bool, float]:
    """客观测试任务确定性规则裁决：支持 choice（选项匹配）与 fill（精确文本匹配）。"""
    exp = expected_answer.strip()
    usr = user_answer.strip()
    if not usr:
        return False, 0.0

    if q_type == "choice":
        norm_exp = _normalize_choice(exp)
        norm_usr = _normalize_choice(usr)
        ok = norm_exp == norm_usr or exp.lower() == usr.lower()
        return ok, (1.0 if ok else 0.0)

    # 精确匹配任务：忽略大小写、首尾标点与多余空白
    clean_exp = re.sub(r"\s+", " ", exp.lower()).strip("。.!！?？,，;；")
    clean_usr = re.sub(r"\s+", " ", usr.lower()).strip("。.!！?？,，;；")
    ok = clean_exp == clean_usr
    return ok, (1.0 if ok else 0.0)
