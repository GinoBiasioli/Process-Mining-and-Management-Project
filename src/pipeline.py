
from typing import Dict
import os
import pandas as pd

from config import RunConfig
from reporting import (
    header, as_series, print_prediction_comparison,
    plot_decision_tree_figure, print_full_tree_rules
)

from EDA import run_eda
from encoding import (
    build_prefix_table,
    make_activity_universe_from_prefix_table,
    boolean_encode_prefix_table,
)
from model import train_and_evaluate_decision_tree
from recommendations import extract_recommendations, evaluate_recommendations

try:
    from recommendations import export_recommendations_xlsx
except Exception:
    export_recommendations_xlsx = None


def maybe_run_eda(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: RunConfig) -> None:
    if not cfg.do_eda:
        return

    header("EDA (display only)")
    run_eda(
        train_df, "train",
        results_dir=str(cfg.project_root),
        case_col=cfg.case_col, act_col=cfg.act_col, time_col=cfg.time_col, label_col=cfg.label_col,
        top_n=cfg.eda_top_n,
        save_plots=False,
    )
    run_eda(
        test_df, "test",
        results_dir=str(cfg.project_root),
        case_col=cfg.case_col, act_col=cfg.act_col, time_col=cfg.time_col, label_col=cfg.label_col,
        top_n=cfg.eda_top_n,
        save_plots=False,
    )


def run_for_k(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: RunConfig, k: int) -> Dict:
    header(f"PREFIX + BOOLEAN ENCODING + MODEL + RECOMMENDATIONS — k={k}")

    # 1) Build prefix tables
    prefix_train = build_prefix_table(
        train_df, prefix_len=k,
        case_col=cfg.case_col, act_col=cfg.act_col, time_col=cfg.time_col, label_col=cfg.label_col,
        pad_token=cfg.pad_token,
    )
    prefix_test = build_prefix_table(
        test_df, prefix_len=k,
        case_col=cfg.case_col, act_col=cfg.act_col, time_col=cfg.time_col, label_col=cfg.label_col,
        pad_token=cfg.pad_token,
    )

    # 2) Activity universe from train only
    universe = make_activity_universe_from_prefix_table(prefix_train, prefix_len=k, pad_token=cfg.pad_token)

    # 3) Boolean encoding
    X_train, y_train, _ = boolean_encode_prefix_table(
        prefix_train, prefix_len=k,
        activity_universe=universe,
        case_col=cfg.case_col, label_col=cfg.label_col,
        pad_token=cfg.pad_token,
    )
    X_test, y_test, _ = boolean_encode_prefix_table(
        prefix_test, prefix_len=k,
        activity_universe=universe,
        case_col=cfg.case_col, label_col=cfg.label_col,
        pad_token=cfg.pad_token,
    )

    print("Prefix tables shape:", prefix_train.shape, prefix_test.shape)
    print("X shapes:", X_train.shape, X_test.shape)

    # 4) Train + evaluate
    result = train_and_evaluate_decision_tree(
        X_train, y_train, X_test, y_test,
        tune=cfg.tune_model,
        scoring=cfg.scoring,
        cv=cfg.cv_folds,
        use_target_recall_threshold=cfg.use_target_recall_threshold,
        target_recall_slow=cfg.target_recall_slow,
    )

    if getattr(result, "best_params", None) is not None:
        print("\nBest params:", result.best_params)

    print_prediction_comparison(
        result.metrics, k,
        plot_cms=cfg.plot_confusion_matrices,
        show_reports=cfg.show_classification_reports
    )

    best_dt = result.model

    # 5) Recommendations
    case_ids_test = prefix_test[cfg.case_col].reset_index(drop=True)
    y_pred_for_recs = (
        result.y_pred_thresholded
        if getattr(result, "y_pred_thresholded", None) is not None
        else result.y_pred_default
    )
    y_pred_for_recs = as_series(y_pred_for_recs, name="y_pred")

    recs = extract_recommendations(
        tree=best_dt,
        X_prefix=X_test.reset_index(drop=True),
        case_ids=case_ids_test.reset_index(drop=True),
        class_values=["true", "false"],
        positive_class=1,   # FAST / true
        y_pred_override=y_pred_for_recs,
    )

    # Optional export
    if cfg.save_recs_xlsx:
        if export_recommendations_xlsx is None:
            print("export_recommendations_xlsx not available (import failed).")
        else:
            cfg.results_dir.mkdir(parents=True, exist_ok=True)
            xlsx_path = export_recommendations_xlsx(
                test_df=test_df,
                case_col=cfg.case_col,
                label_col=cfg.label_col,
                case_ids=case_ids_test,
                y_pred=y_pred_for_recs,
                recommendations=recs,
                k=k,
                output_dir=str(cfg.results_dir),
                sort_col=cfg.time_col,
            )
            print("Saved:", xlsx_path)

    # 6) Evaluate recommendations
    recc_metrics = evaluate_recommendations(
        test_df=test_df,
        case_col=cfg.case_col,
        act_col=cfg.act_col,
        label_col=cfg.label_col,
        recommendations=recs,
        positive_label="true",
        negative_label="false",
        prefix_len=k,
        sort_col=cfg.recc_sort_col,
    )

    print("\n=== RECOMMENDATION METRICS ===")
    ordered_keys = ["TP", "TN", "FP", "FN", "precision_recc", "improvement", "coverage_given_negative_pred"]
    remaining = [kk for kk in recc_metrics.keys() if kk not in ordered_keys]
    ordered_keys += sorted(remaining)
    for kk in ordered_keys:
        if kk in recc_metrics:
            print(f"{kk}: {recc_metrics[kk]}")

    # small sample of non-empty recs
    print(f"\n=== SAMPLE NON-EMPTY RECOMMENDATIONS (up to {cfg.show_recs_sample}) ===")
    shown = 0
    for cid, rec in recs.items():
        if not rec:
            continue

        conds = sorted(list(rec))
        if cfg.max_conds_per_rec and len(conds) > cfg.max_conds_per_rec:
            shown_conds = conds[:cfg.max_conds_per_rec] + [f"... (+{len(conds) - cfg.max_conds_per_rec} more)"]
        else:
            shown_conds = conds

        print(f"Case {cid}: {shown_conds}")
        shown += 1
        if cfg.show_recs_sample and shown >= cfg.show_recs_sample:
            break
    if shown == 0:
        print("No non-empty recommendations found.")

    # 7) Tree plot + optional full rules
    if cfg.plot_decision_tree:
        plot_decision_tree_figure(best_dt, X_train.columns)

    if cfg.print_full_tree:
        if (not cfg.print_full_tree_only_last_k) or (k == cfg.k_values[-1]):
            header(f"FULL TREE RULES — k={k}")
            print_full_tree_rules(best_dt, X_train.columns)

    # Return anything you might want later (tables, saving, etc.)
    return {
        "k": k,
        "result": result,
        "recc_metrics": recc_metrics,
        "n_test_prefixes": len(prefix_test),
    }


def run_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: RunConfig):
    maybe_run_eda(train_df, test_df, cfg)

    outputs = []
    for k in cfg.k_values:
        out = run_for_k(train_df, test_df, cfg, k)
        outputs.append(out)

    print("\nDone.")
    return outputs
