# Nolansbore Project

This repository contains notebooks and helper scripts for modeling spectral data to predict TotalREE (total rare earth elements). The code demonstrates three modeling approaches with interval-aware, group-based cross-validation and includes interpretability analyses.

**Files of interest**

- `notebook/lightgbm.ipynb` — LightGBM workflow: data preparation, group-aware cross-validation, model training, bag-level aggregation, plotting, and SHAP interpretation.
- `notebook/plsr.ipynb` — Partial Least Squares Regression (PLSR) workflow with VIP analysis for variable importance.
- `notebook/multiple-instance-learning.ipynb` — Multiple-Instance Learning (MIL) model with attention pooling implemented in PyTorch (bag-level predictions, interpretability via attention/saliency).
- `add_interval_column.py` — CLI utility that adds an `interval` column to a CSV based on depth values and predefined intervals.

**Requirements**

A minimal Python environment is required. Use the provided `requirements.txt` to install dependencies.

- Python 3.8+
- See `requirements.txt` for package versions (recommended).

**Quick setup (PowerShell)**

```powershell
# Create a virtual environment
python -m venv .venv
# Activate
.\.venv\Scripts\Activate.ps1
# Upgrade pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Preparing your data**

1. The spectral CSV must contain:

   - A depth column (default name: `Depth`).
   - Optionally an `interval` or group column. If missing, use `add_interval_column.py`.
   - Spectral feature columns named as numeric strings (e.g., `400`, `401.5`, ...). The notebooks auto-detect spectral columns.

2. Add an `interval` column (default behavior):

```powershell
python add_interval_column.py hullq.csv
# Output: hullq_interval.csv (by default)
```

3. Use custom intervals (JSON file with [[start,end], ...]):

```powershell
python add_interval_column.py hullq.csv --intervals-json my_intervals.json
```

**Running the notebooks**

Open the notebooks in VS Code or Jupyter and run cells sequentially. Recommended order:

1. `notebook/lightgbm.ipynb`

   - Confirm `DATA_CSV`, `FOLDS_JSON`, and `RANDOM_STATE` in the configuration cell.
   - If `folds_groupkfold.json` is missing, the notebook will generate and save folds.
   - Outputs: OOF predictions, scatter/profile plots, SHAP figures.

2. `notebook/multiple-instance-learning.ipynb`

   - MIL (PyTorch) uses an attention pooling model. Check `RANDOM_STATE` and device (`CPU`/`CUDA`) settings in the config.
   - This notebook contains seeding for reproducibility (NumPy, Python `random`, and PyTorch seeds). See the Reproducibility section below.

3. `notebook/plsr.ipynb`

   - PLSR training and VIP importance scoring. Also honours `RANDOM_STATE` for deterministic behavior where applicable.

**Reproducibility**

To reproduce identical results across runs, follow these guidelines:

- Keep `RANDOM_STATE` fixed in each notebook's configuration cell. All notebooks include a `RANDOM_STATE` constant (default `42`).
- For the MIL (PyTorch) notebook, seeding covers:
  - Python's `random` module (`random.seed(RANDOM_STATE)`).
  - NumPy (`np.random.seed(RANDOM_STATE)`).
  - PyTorch CPU/GPU (`torch.manual_seed`, `torch.cuda.manual_seed_all`).
  - `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` (may reduce performance).
- If you use PyTorch `DataLoader`, set `num_workers=0` for strict reproducibility, or provide a `worker_init_fn` that seeds each worker.
- LightGBM and scikit-learn use NumPy's RNG and their own `random_state` parameters; ensure model constructors set `random_state=RANDOM_STATE` when applicable.

Quick verification: run a notebook twice (fresh kernel each time). Compare saved OOF predictions (or CSV outputs). They should be identical when `RANDOM_STATE` and device settings are unchanged.

**Notes & Tips**

- SHAP on large datasets can be slow and memory-intensive; notebooks subsample background points for SHAP computation.
- Enabling full determinism (`cudnn.deterministic=True`) may degrade training speed. Only enforce it when exact reproducibility is required for experiments.
- If you modify column names, update `DEPTH_COL`, `GROUP_COL`, and path constants in the notebook configuration cells.

**Summary & Next Steps**

- The notebooks are organized for reproducible experiments across modeling approaches (LightGBM, PLSR, MIL).
- Next recommended improvements: add unit tests for helper functions (`depth_to_interval_label`), add CI to validate notebooks run headlessly, and include small example datasets for quick smoke-tests.

**Contact / Contributions**

Open an issue or create a pull request for changes, improvements, or clarifications.

_Last updated: 2025-11-12_
