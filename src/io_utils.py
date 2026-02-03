# io_utils.py
from pathlib import Path
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer


def load_xes_to_log_and_df(xes_path: Path):
    """Load XES as (EventLog, DataFrame)."""
    event_log = xes_importer.apply(str(xes_path))
    df = pm4py.convert_to_dataframe(event_log)
    return event_log, df


def print_basic_sanity(train_log, test_log, train_df, test_df, case_col: str):
    print("\n" + "=" * 90)
    print("Loaded logs")
    print("=" * 90)
    print("EventLog types:", type(train_log), type(test_log))
    print("Train traces:", len(train_log), " | Test traces:", len(test_log))
    print("Train events:", len(train_df), " | Test events:", len(test_df))

    first_case = train_df[case_col].iloc[0]
    print("\nFirst case id:", first_case)
    print("First 3 events (train_df):")
    print(train_df[train_df[case_col] == first_case].head(3))
