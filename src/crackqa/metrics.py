"""Binary classification metrics. Undefined ratios (0 / 0) are reported as 0."""

from __future__ import annotations

import numpy as np


def confusion(y_true, y_pred) -> dict:
    t = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    return {
        "tp": int(np.sum(t & p)),
        "fp": int(np.sum(~t & p)),
        "fn": int(np.sum(t & ~p)),
        "tn": int(np.sum(~t & ~p)),
    }


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def average_precision(y_true, scores) -> float | None:
    """Area under the precision-recall curve (step-wise, as average precision). None without
    positives."""
    t = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    if not t.any():
        return None
    order = np.argsort(-s, kind="stable")
    t, s = t[order], s[order]
    # one point per distinct score
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(t)[last]
    fp = np.cumsum(~t)[last]
    precision = tp / (tp + fp)
    recall = tp / t.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def best_threshold(y_true, scores) -> tuple[float, float]:
    """(threshold, F1) maximising F1 for the rule score >= threshold. Ties keep the highest
    threshold."""
    t = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="stable")
    t, s = t[order], s[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(t)[last]
    fp = np.cumsum(~t)[last]
    fn = t.sum() - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(tp)), where=denom > 0)
    i = int(np.argmax(f1))
    return float(s[last[i]]), float(f1[i])


def summarize(y_true, scores, threshold: float) -> dict:
    """Everything the reports need at one threshold."""
    y_true = np.asarray(y_true, dtype=bool)
    pred = np.asarray(scores, dtype=float) >= threshold
    c = confusion(y_true, pred)
    out = {"threshold": threshold, **c, **prf(c["tp"], c["fp"], c["fn"])}
    out["pr_auc"] = average_precision(y_true, scores)
    out["n"] = int(len(y_true))
    out["positives"] = int(y_true.sum())
    return out
