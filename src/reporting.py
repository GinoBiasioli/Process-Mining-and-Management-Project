
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree


def header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def as_series(x, name: str = "pred") -> Optional[pd.Series]:
    if x is None:
        return None
    if isinstance(x, pd.Series):
        return x.reset_index(drop=True)
    if isinstance(x, pd.DataFrame) and x.shape[1] == 1:
        return x.iloc[:, 0].reset_index(drop=True)
    return pd.Series(list(x), name=name).reset_index(drop=True)


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_cm_2x2(cm, title: str) -> None:
    cm = np.asarray(cm)
    if cm.shape != (2, 2):
        print(f"{title}: {cm}")
        return

    print(f"\n{title}")
    print("".ljust(16) + "pred SLOW".rjust(10) + "pred FAST".rjust(10))
    print("true SLOW".ljust(16) + f"{cm[0, 0]:>10d}{cm[0, 1]:>10d}")
    print("true FAST".ljust(16) + f"{cm[1, 0]:>10d}{cm[1, 1]:>10d}")


def plot_confusion(cm, title: str) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["SLOW (0)", "FAST (1)"])
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def print_prediction_comparison(metrics: Dict, k: int, plot_cms: bool, show_reports: bool) -> None:
    pred = metrics.get("prediction_eval", {})
    if not pred:
        print("No prediction_eval metrics found.")
        return

    cols = ["default"] + (["thresholded"] if "thresholded" in pred else [])

    rows = [
        "accuracy",
        "balanced_accuracy",
        "precision_fast", "recall_fast", "f1_fast",
        "precision_slow", "recall_slow", "f1_slow",
        "f1_macro", "f1_weighted",
        "n_pred_fast", "n_pred_slow",
    ]

    print(f"\n=== PREDICTION METRICS (COMPARABLE) — k={k} ===")
    head = "metric".ljust(22) + "".join(c.ljust(16) for c in cols)
    print(head)
    print("-" * len(head))
    for r in rows:
        line = r.ljust(22)
        for c in cols:
            line += fmt(pred[c].get(r)).ljust(16)
        print(line)

    print(f"\n=== CONFUSION MATRICES — k={k} ===")
    print("(rows=true [SLOW, FAST], cols=pred [SLOW, FAST])")
    for c in cols:
        cm = pred[c].get("confusion_matrix")
        print_cm_2x2(cm, title=c)
        if plot_cms and cm is not None:
            plot_confusion(cm, title=f"Confusion matrix ({c}) — k={k}")

    if show_reports:
        for c in cols:
            rep = pred[c].get("classification_report")
            if rep:
                print(f"\n--- classification_report: {c} (k={k}) ---")
                print(rep)


def plot_decision_tree_figure(dt, feature_names):
    plt.figure(figsize=(24, 12))
    plot_tree(
        dt,
        feature_names=list(feature_names),
        class_names=["SLOW(false)", "FAST(true)"],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def print_full_tree_rules(dt, feature_names):
    """Print full tree as IF-THEN rules (can be very long)."""
    t = dt.tree_
    fn = list(feature_names)

    def walk(node=0, conds=None):
        if conds is None:
            conds = []

        left, right = t.children_left[node], t.children_right[node]

        # leaf
        if left == right:
            counts = t.value[node][0]  # [n_class0, n_class1]
            pred = int(np.argmax(counts))
            total = counts.sum()
            proba_fast = float(counts[1] / total) if total else 0.0

            pred_lbl = "FAST(true)" if pred == 1 else "SLOW(false)"
            print("IF " + " AND ".join(conds) if conds else "IF (root)")
            print(f"  THEN {pred_lbl} | counts={counts.astype(int).tolist()} | P(FAST)={proba_fast:.3f}\n")
            return

        feat = fn[t.feature[node]]
        thr = float(t.threshold[node])

        walk(left,  conds + [f"{feat} <= {thr:.3f}"])
        walk(right, conds + [f"{feat} >  {thr:.3f}"])

    walk()
