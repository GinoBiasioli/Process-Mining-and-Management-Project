# Prefix-Based Predictive Monitoring and Decision-Tree Recommendations for Manufacturing Cycle-Time Outcomes

This repository contains a full, reproducible pipeline for **prefix-based predictive monitoring** on a manufacturing event log (XES), plus a **decision-tree-based recommendation extraction** mechanism.

At a high level, the pipeline:

1. Loads train/test XES logs.
2. Builds **k-prefix** case-level datasets.
3. Encodes prefixes into **boolean activity-presence features**.
4. Trains a **Decision Tree** (optionally tuned with cross-validation).
5. Evaluates prediction performance under:

   * the default decision rule (≈ 0.5), and
   * an optional **thresholding policy** calibrated to reach a target recall for **SLOW**.
6. For prefixes predicted **SLOW**, extracts **actionable recommendations** from compatible FAST paths in the tree, and evaluates them.

---

## Repository structure

```
.
├── src/                # All project code (pipeline modules)
├── data/               # Input event logs (.xes)
├── results/            # Outputs (e.g., exported .xlsx recommendations)
├── .gitignore
└── README.md
```

### `data/`

By default, the code expects these files (relative to project root):

* `data/Production_avg_dur_training_0-80.xes`
* `data/Production_avg_dur_testing_80-100.xes`

If your filenames differ, update them in `src/config.py`.

### `results/`

When enabled, recommendation files are exported as:

* `results/recommendations_k{k}.xlsx`

---

## Requirements

Core dependencies:

* `pm4py`
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `openpyxl` (for `.xlsx` export)

---

## Setup

Clone the repository and install dependencies:

```bash
git clone <YOUR_REPO_URL>
cd Process-Mining-and-Management-Project

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

> If you don’t have a `requirements.txt` yet, you can create one (after installing packages) with:
>
> ```bash
> pip freeze > requirements.txt
> ```

---

## Run the project

Run from the **project root**:

```bash
python src/main.py
```

`src/main.py` is the orchestrator: it loads logs and runs the experiment loop over the configured prefix lengths `k`.

---

## Configuration (important)

All key settings live in:

* `src/config.py` (`RunConfig`)

Common parameters you may want to change:

### Prefix lengths

```python
k_values = [3, 5, 8, 10, 15]
```

(You can run a single k for quick tests.)

### Thresholding policy (optional)

The project supports a thresholding policy that selects a cutoff on **P(SLOW)** to achieve a target recall for SLOW (computed on training data via out-of-fold CV probabilities).

```python
use_target_recall_threshold = True
target_recall_slow = 0.7
```

### Model tuning

```python
tune_model = True
scoring = "recall_slow"
cv_folds = 10
```

### Recommendations export

```python
save_recs_xlsx = True
results_dirname = "results"
```

### Optional plots / verbosity

```python
do_eda = False
plot_confusion_matrices = False
plot_decision_tree = False
print_full_tree = False
```

---

## What the pipeline prints / produces

### Prediction outputs

For each `k`, the console output includes:

* comparable metrics (accuracy, balanced accuracy, per-class precision/recall/f1)
* confusion matrices for:

  * default decision rule
  * thresholded policy (if enabled)
* optional classification reports

**Label convention used internally**

* FAST  → `1` (often stored as `"true"` in the log)
* SLOW  → `0` (often stored as `"false"` in the log)

---

## Recommendation logic (what it means)

Recommendations are extracted **only** for prefixes predicted **SLOW** (negative outcome).

The method:

* scans the decision tree to find **compatible** root→leaf paths that end in a **FAST** leaf,
* picks the “best” compatible FAST path (favoring higher FAST probability + simpler guidance),
* translates the path constraints into human-readable recommendations such as:

  * `Activity X has to be executed`
  * `Activity Y does not have to be executed`

### Recommendation evaluation

Recommendations are evaluated:

* only when a recommendation was actually available for a negatively predicted prefix, and
* only on the **suffix after the prefix** (the future part the recommendation can influence).

Reported metrics include:

* `precision_recc`
* `coverage_given_negative_pred`
* `improvement` (a simple lift-style diagnostic)

### Excel export (`results/`)

When enabled, each exported file `recommendations_k{k}.xlsx` includes one row per test case with:

* case id
* ground truth
* prediction
* recommendation text (if any)
* recommendation type (none / empty / available)
* number of recommendation items
* k

---

## Troubleshooting

### 1) “File not found” for XES logs

Ensure the expected files exist in `data/`, or update:

* `train_relpath`
* `test_relpath`
  in `src/config.py`.

### 2) Import errors when running

Make sure you run from the **project root**:

```bash
python src/main.py
```

### 3) `.xlsx` export issues

Make sure `openpyxl` is installed:

```bash
pip install openpyxl
```

---

## Notes on reproducibility

* The decision tree uses a fixed `random_state`.
* Cross-validation is stratified for stable PR/threshold estimation.



Project developed by **Gino Biasioli**.
