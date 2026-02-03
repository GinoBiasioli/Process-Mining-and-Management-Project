
"""
main.py (orchestrator)

What it does:
- Loads train/test logs
- Runs the experiment loop over k via pipeline.run_experiment()

All heavy logic (metrics printing, plotting, rec extraction/eval) lives in:
- pipeline.py
- reporting.py
- io_utils.py
- config.py
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path (so imports work when running from project root)
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import RunConfig
from io_utils import load_xes_to_log_and_df, print_basic_sanity
from pipeline import run_experiment


def main():
    cfg = RunConfig()

    # Load logs
    train_log, train_df = load_xes_to_log_and_df(cfg.train_path)
    test_log, test_df = load_xes_to_log_and_df(cfg.test_path)

    print_basic_sanity(train_log, test_log, train_df, test_df, case_col=cfg.case_col)

    # Run
    run_experiment(train_df, test_df, cfg)


if __name__ == "__main__":
    main()