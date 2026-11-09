import pytest

from crackqa.metrics import average_precision, best_threshold, confusion, prf, summarize


def test_confusion_and_prf_match_hand_counts():
    y_true = [1, 1, 1, 0, 0, 0, 0, 1]
    y_pred = [1, 1, 0, 1, 0, 0, 0, 0]
    c = confusion(y_true, y_pred)
    assert c == {"tp": 2, "fp": 1, "fn": 2, "tn": 3}
    m = prf(c["tp"], c["fp"], c["fn"])
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 4)
    assert m["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))


def test_no_positives_edge_case():
    m = summarize([0, 0, 0], [0.9, 0.1, 0.2], threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (0, 1, 0, 2)
    assert (m["precision"], m["recall"], m["f1"]) == (0.0, 0.0, 0.0)
    assert m["pr_auc"] is None
    assert prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_average_precision_hand_example():
    # ranked: 1 (P=1, R=.5), 0, 1 (P=2/3, R=1) -> AP = .5 * 1 + .5 * 2/3
    assert average_precision([1, 0, 1], [0.9, 0.8, 0.7]) == pytest.approx(0.5 + 1 / 3)
    assert average_precision([1, 1, 0], [0.9, 0.8, 0.1]) == pytest.approx(1.0)


def test_average_precision_groups_tied_scores():
    assert average_precision([1, 0], [0.5, 0.5]) == pytest.approx(0.5)


def test_best_threshold_picks_the_f1_peak():
    threshold, f1 = best_threshold([0, 0, 1, 1, 0, 1], [0.1, 0.2, 0.6, 0.7, 0.65, 0.9])
    assert threshold == pytest.approx(0.6)
    assert f1 == pytest.approx(2 * 3 / (2 * 3 + 1))
