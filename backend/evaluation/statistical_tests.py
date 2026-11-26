"""Small, dependency-free statistics used to compare an effect against a control distribution."""

from __future__ import annotations

import math


def summarize(xs: list[float]) -> dict:
    n = len(xs)
    if n == 0:
        return {"n": 0}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    t = _t975(n - 1) if n > 1 else float("nan")
    return {"n": n, "mean": m, "std": sd, "min": min(xs), "max": max(xs),
            "ci95_mean": [m - t * se, m + t * se] if n > 1 else None}


def z_score(x: float, controls: list[float]) -> float | None:
    s = summarize(controls)
    if s["n"] < 2 or s["std"] == 0:
        return None
    return (x - s["mean"]) / s["std"]


def empirical_percentile(x: float, controls: list[float]) -> float | None:
    """Percent of controls strictly below x (0 = x is the smallest)."""
    if not controls:
        return None
    return 100.0 * sum(1 for c in controls if c < x) / len(controls)


def _t975(df: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
             10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}
    if df in table:
        return table[df]
    keys = sorted(table)
    for k in keys:
        if k > df:
            return table[k]
    return 1.96
