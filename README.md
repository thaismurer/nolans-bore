# Nolansbore Project

This repository contains notebooks and helper scripts for modeling spectral data to predict TotalREE (total rare earth elements) using LightGBM and PLSR. The notebooks are organized for reproducible, group-aware cross validation (group = depth interval) and include interpretation with SHAP and VIP.

Files of interest

- `lightgbm.ipynb` — LightGBM workflow: data preparation, group-aware CV, training, bag-level aggregation, plotting, SHAP interpretation.
- `plsr.ipynb` — PLS regression workflow with VIP analysis for variable importance.
- `add_interval_column.py` — CLI script that adds an `interval` column to a CSV based on depth values and predefined intervals.

Requirements

A minimal Python environment with the packages used in the notebooks. A `requirements.txt` is provided, but you can install the packages manually if preferred.

- Python 3.8+
- Packages: see `requirements.txt`

Quick setup (PowerShell)

```powershell
# Create a virtual environment
python -m venv .venv
# Activate
.\.venv\Scripts\Activate.ps1
# Install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt
```

How to prepare your data

1. Your spectral CSV must contain:
   - A depth column (default name: `Depth`)
   - A group/interval column (optional). If missing, use `add_interval_column.py` to create it.
   - Spectral columns named with numeric strings (e.g., `400`, `401.5`, ...). These will be detected automatically.

2. To add the `interval` column based on depth using the default intervals from the notebook:

```powershell
python add_interval_column.py hullq.csv
# produces hullq_interval.csv (by default)
```

3. To provide custom intervals (JSON file containing a list of [start, end] pairs):

```powershell
python add_interval_column.py hullq.csv --intervals-json my_intervals.json
```

Running the notebooks

Open the notebooks in VS Code or Jupyter and run cells in order. Recommended sequence:

1. `lightgbm.ipynb`
   - Confirm `DATA_CSV` and `FOLDS_JSON` paths in the configuration cell.
   - If `folds_groupkfold.json` does not exist, the notebook will generate and save deterministic folds (based on `GROUP_COL`).
   - The notebook saves plots (e.g., `lgbm_oof_profile.png`, `lgbm_oof_scatter.png`) and SHAP images.

2. `plsr.ipynb` (if you want PLSR analysis and VIP scores)

Notes and tips

- Reproducibility: the notebooks perform deterministic sorting before fold creation and use a `RANDOM_STATE` variable; keep it fixed for reproducible results.
- Large datasets and SHAP: SHAP can be slow; the notebooks subsample background points when computing SHAP values.
- If you change column names, update `DEPTH_COL` and `GROUP_COL` in the config cell.

Development & testing

You can run `add_interval_column.py` on sample CSV files to quickly create the required `interval` column before running the notebooks.

Contact / Questions

If you want changes to the notebooks (more explanation, added tests, auto-generation of plots into a report, etc.), open an issue or ask for the change and I can implement it.
