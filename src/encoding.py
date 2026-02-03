# -*- coding: utf-8 -*-
"""

Lightweight utilities to transform an **event-level** process log into a
**case-level** dataset suitable for prefix-based predictive monitoring.

What this file does
-------------------
1) Build a *prefix table* with exactly one row per case:
   - case id
   - case label
   - activity at position 1..k (padded if the case trace is shorter than k)

2) Convert the prefix table into a *boolean/presence* feature matrix:
   - For each activity X: create a feature `act=X` ∈ {0,1}
   - `act=X = 1` iff activity X appears at least once in the first k events

Why these choices are useful
----------------------------
- Prefix tables make it explicit what information is available at prediction time.
- Presence encoding is simple, interpretable, and works well with decision trees.
- Keeping a consistent `activity_universe` across train/test ensures aligned
  feature columns.

@author: ginob
"""

import pandas as pd
from typing import List, Tuple, Optional

# -----------------------------------------------------------------------------
# Default column names (compatible with common PM4Py/XES conventions)
# -----------------------------------------------------------------------------
CASE_COL_DEFAULT = "case:concept:name"
ACT_COL_DEFAULT = "concept:name"
TIME_COL_DEFAULT = "time:timestamp"
LABEL_COL_DEFAULT = "case:label"

# Token used to pad prefixes for traces shorter than k.
PAD_TOKEN_DEFAULT = "__PAD__"


def build_prefix_table(
    df: pd.DataFrame,
    prefix_len: int,
    case_col: str = CASE_COL_DEFAULT,
    act_col: str = ACT_COL_DEFAULT,
    time_col: str = TIME_COL_DEFAULT,
    label_col: str = LABEL_COL_DEFAULT,
    pad_token: str = PAD_TOKEN_DEFAULT,
) -> pd.DataFrame:
    """Build a *case-level* prefix dataset with one row per case.

    Output schema
    -------------
    - `case_col`
    - `label_col`
    - `a_1 ... a_prefix_len`
      where `a_i` is the activity name at position i in the case trace.
      If a trace has fewer than `prefix_len` events, remaining positions are
      filled with `pad_token`.

    Prefix definition
    -----------------
    The prefix is the first `prefix_len` events in chronological order.

    Ordering & tie-breaking
    -----------------------
    Events are sorted by (case_id, timestamp). If an `event_nr` column exists,
    it is appended as an additional tie-breaker to keep a stable within-case
    order when timestamps are equal.

    Parameters
    ----------
    df:
        Event-level log as a DataFrame (one row per event).
    prefix_len:
        Number of events in the prefix (k).
    case_col / act_col / time_col / label_col:
        Column names in `df`.
    pad_token:
        Placeholder for missing positions when the trace is shorter than k.

    Returns
    -------
    pd.DataFrame
        One row per case with padded activity positions.
    """

    d = df.copy()

    # Ensure datetime for sorting.
    # - utc=True standardizes timestamps across sources/time zones
    # - errors="coerce" turns invalid timestamps into NaT rather than crashing
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")

    sort_cols = [case_col, time_col]
    if "event_nr" in d.columns:
        sort_cols.append("event_nr")

    d = d.sort_values(sort_cols)

    rows = []

    # groupby(case_col) iterates case by case.
    # sort=False preserves the order produced by the previous sort_values.
    for case_id, g in d.groupby(case_col, sort=False):
        acts = g[act_col].tolist()

        # The case label is typically repeated in every event row.
        # Taking the first is a standard way to get the case-level label.
        y = g[label_col].iloc[0]

        # Keep only the first k events.
        pref = acts[:prefix_len]

        # If the case trace is shorter than k, pad it so every row has the same
        # number of positional columns (a_1..a_k).
        if len(pref) < prefix_len:
            pref = pref + [pad_token] * (prefix_len - len(pref))

        # Build a single output record.
        row = {case_col: case_id, label_col: str(y)}
        for i, a in enumerate(pref, start=1):
            row[f"a_{i}"] = a

        rows.append(row)

    return pd.DataFrame(rows)


def make_activity_universe_from_prefix_table(
    prefix_df: pd.DataFrame,
    prefix_len: int,
    pad_token: str = PAD_TOKEN_DEFAULT,
) -> List[str]:
    """Infer the activity universe (all unique activities) from a prefix table.

    The universe is constructed from `a_1..a_k` columns and excludes `pad_token`.

    Why it matters
    --------------
    The activity universe defines the boolean feature columns `act=<activity>`.
    It should be fixed (learned on train and re-used on test) so train/test have
    aligned feature matrices.

    Returns
    -------
    list[str]
        Sorted list of unique activity names.
    """

    pos_cols = [f"a_{i}" for i in range(1, prefix_len + 1)]

    # values.ravel() flattens the k positional columns into one long array.
    activities = pd.unique(prefix_df[pos_cols].values.ravel())

    # Filter padding and sort for stable column ordering.
    activities = [a for a in activities if a != pad_token]
    return sorted(activities)


def boolean_encode_prefix_table(
    prefix_df: pd.DataFrame,
    prefix_len: int,
    activity_universe: Optional[List[str]] = None,
    case_col: str = CASE_COL_DEFAULT,
    label_col: str = LABEL_COL_DEFAULT,
    pad_token: str = PAD_TOKEN_DEFAULT,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Convert a prefix table into a boolean/presence feature matrix.

    Encoding scheme
    --------------
    For each activity X in `activity_universe`, create a column `act=X` with:
      `act=X = 1` iff X appears at least once in the prefix positions a_1..a_k.

    Parameters
    ----------
    prefix_df:
        Output of `build_prefix_table`.
    prefix_len:
        Prefix length k.
    activity_universe:
        If provided, re-use this exact list to build features.
        If None, it is inferred from `prefix_df` (excluding pad_token).
    label_col:
        Column containing the case label.

    Returns
    -------
    (X, y, activity_universe)
        X:
            DataFrame of shape (n_cases, n_activities).
        y:
            Series of length n_cases (string labels).
        activity_universe:
            The list used to build X columns (returning it helps reuse it).
    """

    d = prefix_df.copy()
    pos_cols = [f"a_{i}" for i in range(1, prefix_len + 1)]

    # If no universe is provided, infer it from the data.
    if activity_universe is None:
        activity_universe = make_activity_universe_from_prefix_table(
            d, prefix_len, pad_token=pad_token
        )

    # Build a set of activities per case row.
    # Using sets makes membership checks fast and the logic easy to read.
    pref_sets = d[pos_cols].apply(lambda r: set(r.values), axis=1)

    # Construct the boolean feature matrix.
    X = pd.DataFrame(index=d.index)
    for act in activity_universe:
        # For each activity, check presence in the row's prefix set.
        X[f"act={act}"] = pref_sets.apply(lambda s: 1 if act in s else 0).astype(int)

    # Labels are returned as strings to keep them consistent with event logs
    # where labels are often stored as string values.
    y = d[label_col].astype(str)

    return X, y, activity_universe

