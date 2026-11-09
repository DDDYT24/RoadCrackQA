import pytest

from crackqa.analysis import bootstrap_f1, confusion_figure, worst_errors


def _rows():
    rows = []
    for s in range(20):
        for t in range(5):
            label = int(t < 2)
            prob = 0.9 if (label and t == 0) or (not label and t == 4 and s % 4 == 0) else 0.1
            rows.append({"source_id": f"s{s}", "label": label, "prob": prob, "baseline_score": 0.5})
    return rows


def test_bootstrap_ci_contains_the_point_estimate():
    r = bootstrap_f1(_rows(), "prob", 0.5, n=300, seed=1)
    # tp 20, fp 5, fn 20 -> F1 = 40 / 65
    assert r["f1"] == pytest.approx(40 / 65)
    lo, hi = r["ci95"]
    assert lo <= r["f1"] <= hi and hi - lo > 0
    assert r["sources"] == 20


def test_bootstrap_is_seeded():
    a = bootstrap_f1(_rows(), "prob", 0.5, n=200, seed=7)
    assert a == bootstrap_f1(_rows(), "prob", 0.5, n=200, seed=7)


def test_worst_errors_order():
    rows = [
        {"label": 0, "prob": 0.7},
        {"label": 0, "prob": 0.95},
        {"label": 1, "prob": 0.2},
        {"label": 1, "prob": 0.01},
        {"label": 1, "prob": 0.9},
    ]
    fp, fn = worst_errors(rows, 0.5, n=5)
    assert [r["prob"] for r in fp] == [0.95, 0.7]
    assert [r["prob"] for r in fn] == [0.01, 0.2]


def test_confusion_figure_writes_a_png(tmp_path):
    path = tmp_path / "fig" / "confusion.png"
    confusion_figure(_rows(), {"model": ("prob", 0.5), "base": ("baseline_score", 0.5)}, path)
    assert path.read_bytes()[:4] == b"\x89PNG"
