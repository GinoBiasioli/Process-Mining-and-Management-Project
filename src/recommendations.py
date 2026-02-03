# -*- coding: utf-8 -*-
"""recommendations.py

Utilities for *extracting* and *evaluating* prescriptive recommendations from a
trained :class:`sklearn.tree.DecisionTreeClassifier` built on **boolean/presence**
features (0/1).

High-level idea
---------------
1) A decision tree is trained to predict an outcome label (e.g., FAST vs SLOW).
2) For prefixes that are predicted **negative** (e.g., predicted SLOW), the tree
   is scanned to find **compatible** root→leaf paths that end in a **positive**
   leaf (e.g., FAST).
3) The "best" compatible positive path is selected and turned into a set of
   actionable constraints at prefix time:
   - "Activity X has to be executed"    (must be present later)
   - "Activity X does not have to be executed" (must remain absent)

Evaluation
----------
Recommendations are evaluated only for cases where a recommendation was produced
(i.e., the case had a negative prediction), and only on the part of the trace
that happens *after* the observed prefix (the only segment the recommendation
can influence).

Implementation notes
--------------------
- This module assumes boolean features that encode **activity presence**.
  Splits in scikit-learn trees usually happen at ~0.5, which aligns with {0,1}.
- "Do not execute" constraints are included if they appear on the chosen positive
  path and are still actionable (the activity has not already occurred).

@author: ginob
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any, Union

import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# =============================================================================
# Internal representation
# =============================================================================

@dataclass(frozen=True)
class PathCondition:
    """A single boolean constraint along a decision-tree path.

    Attributes
    ----------
    feature:
        The feature name used by the model (e.g., "act=Turning & Milling - M8").
    must_be_present:
        True  -> the path goes through the branch meaning feature == 1
        False -> the path goes through the branch meaning feature == 0
    """

    feature: str
    must_be_present: bool


@dataclass
class PositivePath:
    """A root→leaf path ending in a leaf that predicts the positive class."""

    conditions: List[PathCondition]
    proba_true: float  # estimated P(positive_class) at the leaf


# =============================================================================
# Tree -> positive paths
# =============================================================================

def _extract_positive_paths(
    tree: DecisionTreeClassifier,
    feature_names: List[str],
    positive_class: int = 1,
) -> List[PositivePath]:
    """Extract all root→leaf paths that end in a leaf predicting `positive_class`.

    The returned paths are expressed as a list of boolean constraints (0/1) on
    the model features.

    Parameters
    ----------
    tree:
        A fitted scikit-learn DecisionTreeClassifier.
    feature_names:
        Column names of the boolean design matrix used to train/predict.
    positive_class:
        The integer class index considered "positive".

    Returns
    -------
    list of PositivePath
        One entry per positive leaf.
    """

    t = tree.tree_

    def is_leaf(node: int) -> bool:
        # In scikit-learn, a leaf has left==right child.
        return t.children_left[node] == t.children_right[node]

    paths: List[PositivePath] = []

    def dfs(node: int, conds: List[PathCondition]) -> None:
        """Depth-first traversal accumulating path constraints."""
        if is_leaf(node):
            # t.value[node][0] contains class counts at that leaf.
            counts = t.value[node][0]
            total = float(np.sum(counts))
            if total <= 0:
                return

            leaf_pred = int(np.argmax(counts))
            if leaf_pred != positive_class:
                return

            proba_pos = float(counts[positive_class] / total)
            paths.append(PositivePath(conditions=list(conds), proba_true=proba_pos))
            return

        feat_idx = t.feature[node]
        feat_name = feature_names[feat_idx]

        # IMPORTANT: for boolean {0,1} features and threshold ~0.5:
        # - left branch corresponds to feature == 0
        # - right branch corresponds to feature == 1
        left_cond = PathCondition(feature=feat_name, must_be_present=False)
        right_cond = PathCondition(feature=feat_name, must_be_present=True)

        dfs(t.children_left[node], conds + [left_cond])
        dfs(t.children_right[node], conds + [right_cond])

    dfs(0, [])
    return paths


# =============================================================================
# Compatibility + recommendation extraction
# =============================================================================

def _is_path_compatible_with_prefix(
    x_prefix: pd.Series,
    path: PositivePath,
) -> bool:
    """Check whether a positive path is compatible with an observed prefix.

    For boolean presence encoding, incompatibility arises only from constraints
    that require a feature to be 0, while the prefix already has it as 1.

    Intuition
    ---------
    - If the path requires feature==0 but the activity is already present in the
      prefix (feature==1), the path is impossible (past actions can't be undone).
    - If the path requires feature==1 but the prefix has 0, it may still become 1
      later in the suffix, so it remains compatible.
    """
    for c in path.conditions:
        cur = int(x_prefix[c.feature])
        if (not c.must_be_present) and cur == 1:
            return False
    return True


def _recommendable_conditions(
    x_prefix: pd.Series,
    path: PositivePath,
    *,
    include_do_not_execute: bool = True,
) -> List[PathCondition]:
    """Return constraints that are *actionable* at prefix time.

    A path can contain constraints that are already satisfied by the prefix.
    Those are not helpful as recommendations.

    Rules
    -----
    - must_be_present=True: recommend it if currently missing in the prefix (0).
    - must_be_present=False: if enabled, recommend "do not execute" if still
      absent in the prefix (0). If it was present (1), the path would have been
      filtered out as incompatible earlier.

    Note
    ----
    The evaluator treats a recommendation set as **conjunctive**: it is
    considered followed only if *all* included constraints are satisfied.
    """

    recs: List[PathCondition] = []
    for c in path.conditions:
        cur = int(x_prefix[c.feature])

        if c.must_be_present:
            # Missing activity that should appear later
            if cur == 0:
                recs.append(c)
        else:
            # Activity that should remain absent
            if include_do_not_execute and cur == 0:
                recs.append(c)

    return recs


def _condition_to_text(cond: PathCondition) -> str:
    """Human-readable rendering of a PathCondition."""
    act = cond.feature.split("act=", 1)[-1]
    if cond.must_be_present:
        return f"Activity {act} has to be executed"
    else:
        return f"Activity {act} does not have to be executed"


def extract_recommendations(
    tree: DecisionTreeClassifier,
    X_prefix: pd.DataFrame,
    case_ids: pd.Series,
    class_values: List[str] = ["true", "false"],
    positive_class: int = 1,
    y_pred_override: Optional[pd.Series] = None,
    *,
    include_do_not_execute: bool = True,
) -> Dict[Any, Optional[Set[str]]]:
    """Compute recommendations for each prefix.

    Output convention per case:
    - predicted positive  -> None   (no recommendation needed)
    - predicted negative  -> set()  if no compatible positive path exists
    - predicted negative  -> non-empty set of recommendation strings otherwise

    Parameters
    ----------
    tree:
        Fitted DecisionTreeClassifier.
    X_prefix:
        Prefix-level boolean feature matrix.
    case_ids:
        Case identifier for each row in X_prefix.
    positive_class:
        Integer label treated as "positive" by the model.
    y_pred_override:
        Optional external prediction labels (same length as X_prefix).
        Useful when predictions are thresholded outside the tree.
    include_do_not_execute:
        If True, recommendations may include "does not have to be executed"
        constraints derived from the selected positive path.

    Returns
    -------
    dict
        Mapping case_id -> None | set[str]
    """

    feature_names = list(X_prefix.columns)
    pos_paths = _extract_positive_paths(tree, feature_names, positive_class=positive_class)

    # Use the tree predictions unless an override is provided.
    if y_pred_override is None:
        y_pred = pd.Series(tree.predict(X_prefix)).reset_index(drop=True)
    else:
        y_pred = pd.Series(y_pred_override).reset_index(drop=True)

    recs: Dict[Any, Optional[Set[str]]] = {}

    Xp = X_prefix.reset_index(drop=True)
    cids = case_ids.reset_index(drop=True)

    for i in range(len(Xp)):
        cid = cids.iloc[i]
        x = Xp.iloc[i]

        # Predicted positive => recommendation not needed by design.
        if int(y_pred.iloc[i]) == positive_class:
            recs[cid] = None
            continue

        # Predicted negative => find a "best" compatible positive path.
        compatible = [p for p in pos_paths if _is_path_compatible_with_prefix(x, p)]
        if not compatible:
            recs[cid] = set()
            continue

        def score(p: PositivePath) -> Tuple[float, int, int]:
            """Tie-breaker for selecting the best compatible positive path.

            Preference order:
            1) higher estimated P(positive) at the leaf
            2) fewer actionable recommendations (simpler guidance)
            3) shorter overall path (more general)
            """
            rec_items = _recommendable_conditions(
                x, p, include_do_not_execute=include_do_not_execute
            )
            return (p.proba_true, -len(rec_items), -len(p.conditions))

        best = max(compatible, key=score)
        rec_items = _recommendable_conditions(x, best, include_do_not_execute=include_do_not_execute)
        recs[cid] = set(_condition_to_text(c) for c in rec_items)

    return recs


# =============================================================================
# Recommendation evaluation
# =============================================================================

def _canon(s: str) -> str:
    """Canonicalize strings for robust matching (case/space-insensitive)."""
    s = str(s).strip().lower()
    return " ".join(s.split())


def _parse_recommendation_text(rec: str) -> Tuple[str, bool]:
    """Parse a recommendation string back into (activity_name, must_execute).

    Returns
    -------
    (activity_name, must_execute)
        must_execute=True  -> "... has to be executed"
        must_execute=False -> "... does not have to be executed"
    """

    rec = rec.strip()

    if rec.endswith("does not have to be executed"):
        act = (
            rec.replace("Activity", "", 1)
            .replace("does not have to be executed", "")
            .strip()
        )
        return act, False

    if rec.endswith("has to be executed"):
        act = (
            rec.replace("Activity", "", 1)
            .replace("has to be executed", "")
            .strip()
        )
        return act, True

    raise ValueError(f"Unrecognized recommendation format: {rec}")


def evaluate_recommendations(
    test_df: pd.DataFrame,
    case_col: str,
    act_col: str,
    label_col: str,
    recommendations: Dict[Any, Optional[Set[str]]],
    positive_label: str = "true",
    negative_label: str = "false",
    prefix_len: Optional[int] = None,
    sort_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate recommendation quality on a test event log.

    The evaluation is **case-level**:
    - Only cases with negative prediction are considered (recommendations[cid] is not None).
    - If the recommendation set is empty, it is counted as "no recommendation possible"
      and excluded from TP/TN/FP/FN, but included in coverage bookkeeping.

    "Followed" means that **all** constraints are satisfied in the evaluated segment:
      - For each "must execute": the activity appears at least once.
      - For each "must NOT execute": the activity does not appear.

    Segment used to check "followed":
      - If prefix_len is provided: the suffix *after* the prefix (index >= prefix_len).
      - Else: the whole trace (fallback).

    Returned confusion counts:
      - TP: followed & outcome is positive_label
      - TN: not followed & outcome is negative_label
      - FP: followed & outcome is negative_label
      - FN: not followed & outcome is positive_label

    Returns
    -------
    dict
        Counts, core metrics (precision/recall/f1/accuracy), coverage, and a simple
        improvement/lift diagnostic.
    """

    # Normalize recommendations values to either None or a set[str].
    recs_norm: Dict[Any, Optional[Set[str]]] = {}
    for cid, rec in recommendations.items():
        if rec is None:
            recs_norm[cid] = None
        else:
            recs_norm[cid] = set(rec)

    # Ensure stable within-case ordering so prefix slicing is meaningful.
    df = test_df.copy()
    if sort_col is not None and sort_col in df.columns:
        df = df.sort_values([case_col, sort_col], kind="mergesort")
    else:
        # If events are already in order, this preserves it per case.
        df["__row_idx__"] = np.arange(len(df))
        df = df.sort_values([case_col, "__row_idx__"], kind="mergesort")

    # Ground truth outcome label per case (assumes label stored consistently per event).
    gt = df.groupby(case_col)[label_col].first().astype(str)

    # Pre-materialize per-case activity sequences for fast access.
    seqs = df.groupby(case_col)[act_col].apply(lambda s: list(map(str, s.tolist())))

    TP = TN = FP = FN = 0
    n_negative_pred = 0
    n_rec_available = 0
    n_rec_empty = 0

    for cid, rec in recs_norm.items():
        if rec is None:
            # Positive prediction => excluded from recommendation evaluation.
            continue

        n_negative_pred += 1

        if rec == set():
            # Negative prediction but no actionable positive path found.
            n_rec_empty += 1
            continue

        n_rec_available += 1

        y_true = gt.get(cid, None)
        if y_true is None:
            # Case id missing from test_df; skip silently.
            continue

        acts_list = seqs.get(cid, [])
        if prefix_len is None:
            seg_list = acts_list
        else:
            # Evaluate only on suffix after prefix: recommendations cannot affect the past.
            start = min(int(prefix_len), len(acts_list))
            seg_list = acts_list[start:]

        # Using a set makes membership checks O(1) (presence/absence constraints).
        seg_set = set(_canon(a) for a in seg_list)

        # A recommendation is followed only if ALL constraints hold.
        followed = True
        for r in rec:
            act, must = _parse_recommendation_text(r)
            act_c = _canon(act)

            if must and (act_c not in seg_set):
                followed = False
                break
            if (not must) and (act_c in seg_set):
                followed = False
                break

        is_pos = (str(y_true) == positive_label)
        is_neg = (str(y_true) == negative_label)

        if followed and is_pos:
            TP += 1
        elif (not followed) and is_neg:
            TN += 1
        elif followed and is_neg:
            FP += 1
        elif (not followed) and is_pos:
            FN += 1

    # Metrics are computed over the (TP,TN,FP,FN) contingency table.
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0.0

    # Coverage among negatively predicted prefixes:
    # n_rec_available counts only NON-EMPTY recommendation sets.
    coverage = (n_rec_available / n_negative_pred) if n_negative_pred else 0.0

    # Simple improvement (lift) diagnostic:
    #   P(positive | followed) - P(positive | not followed)
    p_pos_followed = TP / (TP + FP) if (TP + FP) else 0.0
    p_pos_not_followed = FN / (FN + TN) if (FN + TN) else 0.0
    improvement = p_pos_followed - p_pos_not_followed

    return {
        # counts
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,

        # core metrics (often reported)
        "precision_recc": precision,
        "improvement": improvement,
        "coverage_given_negative_pred": coverage,

        # additional useful metrics
        "recall_recc": recall,
        "f1_recc": f1,
        "accuracy_recc": acc,
        "p_fast_followed": p_pos_followed,
        "p_fast_not_followed": p_pos_not_followed,

        # volumes / bookkeeping
        "n_negative_pred": n_negative_pred,
        "n_rec_available": n_rec_available,
        "n_rec_empty": n_rec_empty,
        "prefix_len_used": prefix_len,
    }


# =============================================================================
# Excel export helper
# =============================================================================

def _pred_to_label(
    pred: Union[int, float, str],
    positive_label: str,
    negative_label: str,
) -> str:
    """Map predictions to the same string label space as the event log.

    Supports:
    - numeric predictions {0,1}
    - already-labeled strings matching `positive_label` / `negative_label`

    Any other value is returned as a string.
    """

    if pd.isna(pred):
        return ""

    if isinstance(pred, str):
        p = pred.strip().lower()
        if p == str(positive_label).strip().lower():
            return str(positive_label)
        if p == str(negative_label).strip().lower():
            return str(negative_label)
        return pred

    try:
        pi = int(pred)
    except Exception:
        return str(pred)

    if pi == 1:
        return str(positive_label)
    if pi == 0:
        return str(negative_label)
    return str(pi)


def export_recommendations_xlsx(
    *,
    test_df: pd.DataFrame,
    case_col: str,
    label_col: str,
    case_ids: pd.Series,
    y_pred: Union[pd.Series, np.ndarray, list],
    recommendations: Dict[Any, Optional[Set[str]]],
    k: int,
    output_dir: str = "results",
    positive_label: str = "true",
    negative_label: str = "false",
    sort_col: Optional[str] = None,
) -> str:
    """Export a per-case Excel file with ground truth, prediction, and recommendations.

    The exported file is convenient for manual inspection and discussion.

    Parameters
    ----------
    test_df:
        Event-level dataframe for the test split.
    case_col / label_col:
        Column names for case id and outcome label.
    case_ids:
        Case ids aligned with `y_pred` and `recommendations`.
    y_pred:
        Model predictions aligned with `case_ids`.
    recommendations:
        Mapping case_id -> None | set[str] as produced by `extract_recommendations`.
    k:
        Prefix length (stored as a column in the output).
    output_dir:
        Folder where the Excel file will be written.

    Returns
    -------
    str
        Path to the created .xlsx file.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Ensure ordering inside each case (important if label is stored per-event).
    df = test_df.copy()
    if sort_col is not None and sort_col in df.columns:
        df = df.sort_values([case_col, sort_col], kind="mergesort")

    gt = df.groupby(case_col)[label_col].first().astype(str)

    cids = case_ids.reset_index(drop=True)
    y_pred_s = pd.Series(y_pred).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for i, cid in enumerate(cids.tolist()):
        y_true = gt.get(cid, "")

        # Defensive indexing if arrays are misaligned.
        y_hat = y_pred_s.iloc[i] if i < len(y_pred_s) else np.nan
        y_hat_lbl = _pred_to_label(y_hat, positive_label=positive_label, negative_label=negative_label)

        rec = recommendations.get(cid, None)
        if rec is None:
            rec_type = "none_pred_positive"
            rec_txt = ""
            n_items = 0
        else:
            rec_set = set(rec)
            if len(rec_set) == 0:
                rec_type = "empty_no_rec_possible"
                rec_txt = ""
                n_items = 0
            else:
                rec_type = "available"
                # Sorting makes the file stable across runs (helps diffing and reviews).
                rec_txt = "; ".join(sorted(map(str, rec_set)))
                n_items = len(rec_set)

        rows.append(
            {
                "case_id": cid,
                "ground_truth": str(y_true),
                "predicted": str(y_hat_lbl),
                "recommendation": rec_txt,
                "rec_type": rec_type,
                "n_rec_items": n_items,
                "k": int(k),
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, f"recommendations_k{k}.xlsx")
    out_df.to_excel(out_path, index=False, engine="openpyxl")
    return out_path

