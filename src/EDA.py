# -*- coding: utf-8 -*-
"""

Small exploratory data analysis (EDA) helpers for event logs.

The functions in this file operate on an **event-level** dataframe where each
row is an event and cases/traces are identified by a case id column.

Main outputs
------------
- A case-level summary table (one row per case) containing:
  - label (if available)
  - number of events
  - start/end timestamps
  - throughput time (end - start)

- A set of quick sanity-check plots:
  - label distribution (case-level)
  - trace length distribution
  - throughput distribution
  - most frequent activities
  - most frequent start/end activities

- A quick look at the most common variants (unique activity sequences).

Design notes
------------
- Timestamps are parsed with utc=True for consistency.
- Sorting is important when extracting start/end activities and variants.
  If an `event_nr` column exists, it is used as a tie-breaker when timestamps
  are equal.

@author: ginob
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Case-level aggregation
# =============================================================================

def build_case_table(
    df: pd.DataFrame,
    case_col: str,
    time_col: str,
    label_col: str,
) -> pd.DataFrame:
    """Aggregate an event log into a case-level table (one row per trace).

    Parameters
    ----------
    df:
        Event-level dataframe.
    case_col:
        Column identifying the case/trace id.
    time_col:
        Timestamp column.
    label_col:
        Case-level label column (often repeated on each event row). If the column
        is missing, the returned label field will be None.

    Returns
    -------
    pd.DataFrame
        One row per case with basic trace descriptors.
    """

    d = df.copy()

    # Parse timestamps defensively; invalid timestamps become NaT.
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")

    # Group by case. Sorting by time ensures start/end are meaningful.
    g = d.sort_values(time_col).groupby(case_col)

    # Build case-level features.
    cases = pd.DataFrame({
        case_col: g[case_col].first(),
        "label": g[label_col].first() if label_col in d.columns else None,
        "n_events": g.size(),
        "start_time": g[time_col].min(),
        "end_time": g[time_col].max(),
    })

    # Throughput time (duration of the trace).
    cases["throughput_seconds"] = (cases["end_time"] - cases["start_time"]).dt.total_seconds()

    return cases.reset_index(drop=True)


# =============================================================================
# EDA runner
# =============================================================================

def run_eda(
    df: pd.DataFrame,
    name: str,
    results_dir: str,
    case_col: str,
    act_col: str,
    time_col: str,
    label_col: str,
    top_n: int = 20,
    save_plots: bool = False,
) -> pd.DataFrame:
    """Run quick EDA prints + plots for an event log.

    Parameters
    ----------
    df:
        Event-level dataframe.
    name:
        Label used in plot titles and filenames.
    results_dir:
        Where plots are saved if `save_plots=True`.
    case_col / act_col / time_col / label_col:
        Column names.
    top_n:
        Number of most frequent activities shown in the frequency plot.
    save_plots:
        If True, each plot is saved as a PNG in `results_dir`.

    Returns
    -------
    pd.DataFrame
        Case-level table (one row per case).
    """

    os.makedirs(results_dir, exist_ok=True)
    cases = build_case_table(df, case_col=case_col, time_col=time_col, label_col=label_col)

    # -------------------------------------------------------------------------
    # Console summaries
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"EDA — {name}")
    print("=" * 90)
    print("Events (rows):", len(df))
    print("Traces (unique cases):", df[case_col].nunique())
    print("Unique activities:", df[act_col].nunique())

    # Simple missingness profile (top columns).
    print("\nMissingness (top 15):")
    print(df.isna().mean().sort_values(ascending=False).head(15))

    # Case-level label distribution (if labels exist).
    if cases["label"].notna().any():
        print("\nLabel distribution (case-level):")
        print(cases["label"].value_counts(dropna=False))
        print("\nLabel %:")
        print((cases["label"].value_counts(normalize=True, dropna=False) * 100).round(2).astype(str) + "%")

    # Trace length diagnostics (helps understand how often padding happens).
    print("\nTrace length stats (#events per trace):")
    print(cases["n_events"].describe())
    for k in [5, 10]:
        print(f"% traces shorter than {k}: {(cases['n_events'] < k).mean():.2%}")

    # Throughput diagnostics.
    print("\nThroughput time stats (seconds):")
    print(cases["throughput_seconds"].describe())

    # -------------------------------------------------------------------------
    # Plot helper
    # -------------------------------------------------------------------------
    def maybe_save(fig_name: str) -> None:
        """Save current matplotlib figure if enabled."""
        if save_plots:
            plt.savefig(os.path.join(results_dir, fig_name), dpi=200)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    # 1) Label distribution plot (case-level)
    if cases["label"].notna().any():
        ax = cases["label"].value_counts().plot(kind="bar")
        ax.set_title(f"Label distribution — {name}")
        ax.set_xlabel(label_col)
        ax.set_ylabel("# traces")
        plt.tight_layout()
        maybe_save(f"label_dist_{name}.png")
        plt.show()

    # 2) Trace length distribution
    plt.hist(cases["n_events"].values, bins=30)
    plt.title(f"Trace length distribution — {name}")
    plt.xlabel("# events per trace")
    plt.ylabel("# traces")
    plt.tight_layout()
    maybe_save(f"trace_lengths_{name}.png")
    plt.show()

    # 3) Throughput time distribution
    vals = cases["throughput_seconds"].dropna().values
    plt.hist(vals, bins=30)
    plt.title(f"Throughput time distribution — {name}")
    plt.xlabel("throughput (seconds)")
    plt.ylabel("# traces")
    plt.tight_layout()
    maybe_save(f"throughput_{name}.png")
    plt.show()

    # 4) Top activities by event frequency
    vc = df[act_col].value_counts().head(top_n)
    ax = vc.sort_values().plot(kind="barh")
    ax.set_title(f"Top {top_n} activities (event counts) — {name}")
    ax.set_xlabel("# events")
    ax.set_ylabel("activity")
    plt.tight_layout()
    maybe_save(f"top_activities_{name}.png")
    plt.show()

    # -------------------------------------------------------------------------
    # Start/end activities + variants require stable within-case ordering
    # -------------------------------------------------------------------------
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")

    sort_cols = [case_col, time_col]
    if "event_nr" in d.columns:
        # Tie-breaker when timestamps are identical.
        sort_cols.append("event_nr")

    d = d.sort_values(sort_cols)

    # Start activities: first event per case.
    starts = d.groupby(case_col).first()[act_col].value_counts().head(15)
    ax = starts.sort_values().plot(kind="barh")
    ax.set_title(f"Top start activities — {name}")
    plt.tight_layout()
    maybe_save(f"start_activities_{name}.png")
    plt.show()

    # End activities: last event per case.
    ends = d.groupby(case_col).last()[act_col].value_counts().head(15)
    ax = ends.sort_values().plot(kind="barh")
    ax.set_title(f"Top end activities — {name}")
    plt.tight_layout()
    maybe_save(f"end_activities_{name}.png")
    plt.show()

    # Variants: represent each trace as the full sequence of activities.
    # Using tuples makes sequences hashable so value_counts can count them.
    seq = d.groupby(case_col)[act_col].apply(tuple)
    print("\nTop variants (top 10):")
    print(seq.value_counts().head(10))

    # The case table can be exported if needed for inspection.
    # cases.to_csv(os.path.join(results_dir, f"cases_{name}.csv"), index=False)

    return cases

