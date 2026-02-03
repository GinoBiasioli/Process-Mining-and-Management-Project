# -*- coding: utf-8 -*-
"""model_tree.py

Decision tree training + evaluation utilities for prefix-based predictive
monitoring.

This module provides:
- A consistent label normalization step ("true"/"false" -> 1/0).
- A compact, comparable evaluation function for binary predictions.
- Optional hyperparameter tuning via GridSearchCV.
- A cross-validated Precision–Recall (PR) curve for the *SLOW* class, computed
  on training data only using out-of-fold probabilities.
- An optional probability-threshold policy that selects a cutoff on P(SLOW)
  to reach a target recall for SLOW on training-CV, then applies it once on test.

Important conventions
---------------------
- Internal binary coding used here:
    FAST  -> 1   (label string "true")
    SLOW  -> 0   (label string "false")

- For PR/thresholding, the SLOW class is treated as the "positive" class.
  Since the model outputs P(FAST), we compute:
      P(SLOW) = 1 - P(FAST)

Why the CV PR curve matters
---------------------------
PR curves and threshold selection can easily become over-optimistic if they are
computed on the same data used to fit the model. To avoid that, this file uses
*out-of-fold* probabilities from cross_val_predict on the training split.

@author: ginob
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    make_scorer,
)


# =============================================================================
# Prediction evaluation
# =============================================================================

def evaluate_binary_predictions(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> Dict[str, Any]:
    """Compute a consistent set of binary classification metrics.

    Inputs
    ------
    y_true_bin, y_pred_bin:
        Arrays coded as: 1=FAST, 0=SLOW.

    Returns
    -------
    dict
        - Overall metrics: accuracy, balanced_accuracy
        - Per-class metrics for FAST and SLOW: precision/recall/f1
        - Macro and weighted F1 for robustness across class imbalance
        - Confusion matrix and a formatted classification report

    Notes
    -----
    Confusion matrix is built with labels=[0,1] so that the ordering is stable:
      rows = true [SLOW, FAST]
      cols = pred [SLOW, FAST]
    """

    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_pred_bin = np.asarray(y_pred_bin).astype(int)

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    acc = float(accuracy_score(y_true_bin, y_pred_bin))
    bacc = float(balanced_accuracy_score(y_true_bin, y_pred_bin))

    # Per-class metrics returned in the same order as labels=[0,1].
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, labels=[0, 1], zero_division=0
    )

    # Index mapping: 0 -> SLOW, 1 -> FAST
    precision_slow, precision_fast = float(prec[0]), float(prec[1])
    recall_slow, recall_fast = float(rec[0]), float(rec[1])
    f1_slow, f1_fast = float(f1[0]), float(f1[1])

    # classification_report (as dict) is convenient to get macro/weighted values.
    report_dict = classification_report(
        y_true_bin,
        y_pred_bin,
        labels=[0, 1],
        target_names=["SLOW(0)", "FAST(1)"],
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,

        "precision_fast": precision_fast,
        "recall_fast": recall_fast,
        "f1_fast": f1_fast,

        "precision_slow": precision_slow,
        "recall_slow": recall_slow,
        "f1_slow": f1_slow,

        "f1_macro": float(report_dict["macro avg"]["f1-score"]),
        "f1_weighted": float(report_dict["weighted avg"]["f1-score"]),

        # Volumes are often useful to interpret the operating point.
        "n_pred_fast": int(np.sum(y_pred_bin == 1)),
        "n_pred_slow": int(np.sum(y_pred_bin == 0)),

        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true_bin,
            y_pred_bin,
            labels=[0, 1],
            target_names=["SLOW(0)", "FAST(1)"],
            digits=4,
            zero_division=0,
        ),
    }


# =============================================================================
# Label normalization
# =============================================================================

def normalize_labels(y: pd.Series) -> pd.Series:
    """Map string labels to the internal binary coding.

    Expected label strings (case-insensitive):
      - "true"  -> 1  (FAST)
      - "false" -> 0  (SLOW)

    A strict mapping is used so mistakes in the upstream log are caught early.
    """

    s = y.astype(str).str.lower().str.strip()

    # Replace with string digits first (avoids pandas dtype warnings), then cast.
    s = s.replace({"true": "1", "false": "0"})

    bad = s[~s.isin(["0", "1"])]
    if len(bad) > 0:
        raise ValueError(
            f"Unknown labels found (showing up to 10): {bad.unique()[:10]}. "
            "Update normalize_labels() mapping."
        )

    return pd.to_numeric(s, errors="raise").astype(int)


# =============================================================================
# Threshold selection for target recall (SLOW as positive)
# =============================================================================

def threshold_for_target_recall(
    recall: np.ndarray,
    thresholds: np.ndarray,
    target_recall: float,
) -> float:
    """Choose a probability threshold that achieves a target recall.

    Context
    -------
    This is used to choose a cutoff on P(SLOW) based on a PR curve where SLOW is
    treated as the positive class.

    How sklearn returns PR curve arrays
    ----------------------------------
    precision_recall_curve returns:
      - precision:  length = n_thresholds + 1
      - recall:     length = n_thresholds + 1
      - thresholds: length = n_thresholds

    Thresholds align with precision[1:] and recall[1:], so recall[1:] is used.

    Policy implemented
    ------------------
    - If the target recall is unreachable: return the minimum threshold (most
      permissive, maximizes recall).
    - Otherwise: return the *largest* threshold that still satisfies
      recall >= target, i.e. the strictest threshold that meets the recall goal.

    Returns
    -------
    float
        Selected threshold on P(SLOW).
    """

    if thresholds is None or len(thresholds) == 0:
        # Degenerate case: PR curve contains no thresholds.
        return 0.5

    r = recall[1:]

    if not np.any(r >= target_recall):
        return float(thresholds[0])

    idx = np.where(r >= target_recall)[0][-1]
    return float(thresholds[idx])


# =============================================================================
# Result container
# =============================================================================

@dataclass
class ModelResult:
    """Bundle outputs of training + evaluation in one object."""

    model: DecisionTreeClassifier
    best_params: Dict[str, Any]
    metrics: Dict[str, Any]

    # Predictions using sklearn default decision rule (argmax, effectively ~0.5).
    y_pred_default: pd.Series

    # Predictions using a P(SLOW) threshold (if enabled).
    y_pred_thresholded: Optional[pd.Series] = None


# =============================================================================
# Main training function
# =============================================================================

def train_and_evaluate_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    random_state: int = 42,
    tune: bool = True,
    scoring: str = "f1",
    cv: int = 5,
    plot_pr_curve_slow: bool = True,
    # Thresholding policy
    use_target_recall_threshold: bool = True,
    target_recall_slow: float = 0.80,
) -> ModelResult:
    """Train a decision tree and evaluate prediction quality.

    Steps
    -----
    1) Normalize labels to 1=FAST, 0=SLOW.
    2) Optionally tune hyperparameters with GridSearchCV.
    3) Evaluate default predictions on test.
    4) Compute a *training-only* cross-validated PR curve for SLOW.
    5) Optionally select a threshold on P(SLOW) from the training-CV PR curve and
       apply it once on the test probabilities to create thresholded predictions.

    Parameters
    ----------
    X_train, y_train:
        Case-level training features and labels.
    X_test, y_test:
        Case-level test features and labels.
    tune:
        If True, run GridSearchCV; otherwise fit the base tree as-is.
    scoring:
        Scoring used for GridSearchCV. Can be a sklearn scorer string or a custom
        scorer name handled below.
    cv:
        Number of CV folds used for GridSearchCV and for out-of-fold PR estimates.
    plot_pr_curve_slow:
        If True, show the PR curve (training-CV) for the SLOW class.
    use_target_recall_threshold:
        If True, compute a P(SLOW) cutoff based on training-CV PR to reach
        `target_recall_slow`, then apply it on test.

    Returns
    -------
    ModelResult
        Contains the fitted model, params, metrics, and prediction vectors.
    """

    # 1) Normalize labels
    y_train_bin = normalize_labels(y_train).astype(int)
    y_test_bin = normalize_labels(y_test).astype(int)

    base_model = DecisionTreeClassifier(random_state=random_state)

    # 2) Hyperparameter tuning
    if tune:
        param_grid = {
            "max_depth": [3, 5, 7, 8, 10],
            "min_samples_split": [5, 7, 8, 10, 12],
            "min_samples_leaf": [3, 5, 7, 10, 12, 15],
            "criterion": ["gini", "entropy"],
        }

        # Optional custom scoring focused on the SLOW class.
        if scoring == "recall_slow":
            scoring_used = make_scorer(recall_score, pos_label=0)
        elif scoring == "f1_slow":
            scoring_used = make_scorer(f1_score, pos_label=0)
        else:
            scoring_used = scoring

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring_used,
            n_jobs=-1,
        )

        grid.fit(X_train, y_train_bin)
        model = grid.best_estimator_
        best_params = grid.best_params_

    else:
        model = base_model.fit(X_train, y_train_bin)
        best_params = model.get_params()

    # 3) Default test evaluation (argmax rule)
    y_pred_default = pd.Series(model.predict(X_test), index=X_test.index, name="y_pred_default")
    default_eval = evaluate_binary_predictions(y_test_bin.values, y_pred_default.values)

    metrics: Dict[str, Any] = {
        "prediction_eval": {"default": default_eval},
        "n_train_cases": int(len(X_train)),
        "n_test_cases": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "scoring": scoring,
        "cv": cv,
        "tune": tune,
    }

    # -------------------------------------------------------------------------
    # 4) Training-only CV PR curve for SLOW
    # -------------------------------------------------------------------------
    # The tree is trained to predict FAST=1. For SLOW-as-positive PR analysis:
    #   y_slow = 1 - y_fast
    #   p_slow = 1 - p_fast

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Use the final (tuned or untuned) hyperparameters for out-of-fold estimates.
    if tune:
        cv_model = DecisionTreeClassifier(random_state=random_state, **best_params)
    else:
        cv_model = DecisionTreeClassifier(random_state=random_state, **model.get_params())

    # cross_val_predict with method="predict_proba" returns out-of-fold probabilities.
    proba_oof = cross_val_predict(
        cv_model,
        X_train,
        y_train_bin,
        cv=cv_splitter,
        method="predict_proba",
        n_jobs=-1,
    )

    # scikit-learn class order is [0, 1] => column 1 corresponds to P(FAST=1).
    p_fast_oof = proba_oof[:, 1]
    p_slow_oof = 1.0 - p_fast_oof
    y_slow_train = 1 - y_train_bin

    precision, recall, thresholds = precision_recall_curve(y_slow_train, p_slow_oof)
    ap_slow = float(average_precision_score(y_slow_train, p_slow_oof))

    metrics["pr_slow_cv"] = {
        "average_precision": ap_slow,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }

    if plot_pr_curve_slow:
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall (SLOW as positive)")
        plt.ylabel("Precision (SLOW as positive)")
        plt.title(f"PR curve (CV on training) — SLOW class | AP={ap_slow:.4f}")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # 5) Optional threshold selection and thresholded test evaluation
    # -------------------------------------------------------------------------
    y_pred_thresholded: Optional[pd.Series] = None

    if use_target_recall_threshold:
        # Choose tau* on P(SLOW) from training-CV PR curve.
        tau_star = threshold_for_target_recall(recall, thresholds, target_recall_slow)

        # These prints make runs easier to interpret when comparing operating points.
        print(f"TARGET_RECALL_SLOW={target_recall_slow:.3f}  -> tau_star (P(SLOW) cutoff) = {tau_star:.4f}")
        print(f"Equivalent cutoff on P(FAST) is {1 - tau_star:.4f}")

        # Compute test probabilities under the final fitted model.
        p_fast_test = model.predict_proba(X_test)[:, 1]
        p_slow_test = 1.0 - p_fast_test

        # Predict SLOW if P(SLOW) >= tau_star.
        # Here we create a helper vector where 1 means "predicted SLOW".
        y_pred_slow_test = (p_slow_test >= tau_star).astype(int)

        # Convert back to internal coding (FAST=1, SLOW=0):
        # if predicted SLOW (1) => predicted FAST = 0
        y_pred_thr_bin = 1 - y_pred_slow_test

        y_pred_thresholded = pd.Series(y_pred_thr_bin, index=X_test.index, name="y_pred_thresholded")
        thr_eval = evaluate_binary_predictions(y_test_bin.values, y_pred_thresholded.values)

        metrics["threshold_policy"] = {
            "method": "target_recall_slow_on_train_cv",
            "target_recall_slow": float(target_recall_slow),
            "chosen_threshold_p_slow": float(tau_star),
        }

        metrics["prediction_eval"]["thresholded"] = thr_eval

        # Extra view: compute precision/recall for SLOW using SLOW-as-1 coding.
        y_test_slow = 1 - y_test_bin
        prec_slow = float(precision_score(y_test_slow, y_pred_slow_test, zero_division=0))
        rec_slow = float(recall_score(y_test_slow, y_pred_slow_test, zero_division=0))

        cm_thr = confusion_matrix(y_test_bin, y_pred_thresholded)
        report_thr = classification_report(y_test_bin, y_pred_thresholded, digits=4)

        metrics["thresholded_for_slow"] = {
            "precision_slow": prec_slow,
            "recall_slow": rec_slow,
            "confusion_matrix_fast_coding": cm_thr.tolist(),
            "classification_report_fast_coding": report_thr,
        }

    return ModelResult(
        model=model,
        best_params=best_params,
        metrics=metrics,
        y_pred_default=y_pred_default,
        y_pred_thresholded=y_pred_thresholded,
    )



