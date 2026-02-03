
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class RunConfig:
    # ---------- project paths ----------
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    train_relpath: Path = Path("data/Production_avg_dur_training_0-80.xes")
    test_relpath: Path = Path("data/Production_avg_dur_testing_80-100.xes")
    results_dirname: str = "results"

    # ---------- column names ----------
    case_col: str = "case:concept:name"
    act_col: str = "concept:name"
    time_col: str = "time:timestamp"
    label_col: str = "case:label"

    # ---------- experiment settings ----------
    #k_values: List[int] = field(default_factory=lambda: [3, 5, 8, 10, 15])
    k_values: List[int] = field(default_factory=lambda: [5])
    pad_token: str = "__PAD__"

    # ---------- EDA ----------
    do_eda: bool = False
    eda_top_n: int = 20

    # ---------- model tuning ----------
    tune_model: bool = True
    scoring: str = "recall_slow"
    cv_folds: int = 10
    use_target_recall_threshold: bool = True
    target_recall_slow: float = 0.7

    # ---------- prediction reporting ----------
    show_classification_reports: bool = True
    plot_confusion_matrices: bool = False

    # ---------- recommendations ----------
    show_recs_sample: int = 0
    max_conds_per_rec: int = 0
    save_recs_xlsx: bool = True
    recc_sort_col: Optional[str] = "event_nr"   # set None if not used

    # ---------- decision tree output ----------
    plot_decision_tree: bool = False
    print_full_tree: bool = False
    print_full_tree_only_last_k: bool = False

    @property
    def train_path(self) -> Path:
        return self.project_root / self.train_relpath

    @property
    def test_path(self) -> Path:
        return self.project_root / self.test_relpath

    @property
    def results_dir(self) -> Path:
        return self.project_root / self.results_dirname