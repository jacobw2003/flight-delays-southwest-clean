# Southwest Flight Delays – Preprocessing, Analysis, and ML

## Quick Start
- Create environment and install deps:
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
- Run the end-to-end notebook:
```bash
jupyter lab  # or jupyter notebook
# Open and run: Southwest_Flight_Delays_Pipeline.ipynb
```

## Data Layout
- Input CSV (preprocessed from raw): `data/csv_data/features_added_southwest.csv`
- Final preprocessed dataset: `data/preprocessed_data/southwest_final_preprocessed.csv`
- Analysis plots: `analysis/data_analysis/analysis_plots/`

## Preprocessing (CLI)
Generate the final ML-ready dataset and remove COVID years (2020–2021):
```bash
python analysis/preprocessing/complete_preprocessing_pipeline.py
```
This performs:
- Data quality checks (missing, duplicates)
- Redundant column removal and type conversion
- Feature creation: `Year`, `Month`, `Day`, `DayOfWeek`, `Quarter`, `Season`, `Route`, `DelaySeverity`
- Filter 2020–2021
- Writes `data/preprocessed_data/southwest_final_preprocessed.csv`

## Data Analysis (CLI)
Run the consolidated analysis (routes, seasonal patterns, STL decomposition) and save plots:
```bash
python analysis/data_analysis/unified_analysis.py
# Optional: --data-path /absolute/path/to/your.csv
```
Outputs plots to `analysis/data_analysis/analysis_plots/`.

## Machine Learning (CLI)
Unified entrypoint:
```bash
# Realistic (pre-departure features only; no leakage)
python analysis/machine_learning/unified_ml.py realistic

# Consumer two-stage pipeline (classification + regression)
python analysis/machine_learning/unified_ml.py consumer

# Operational pipeline (actionable causes and insights)
python analysis/machine_learning/unified_ml.py operational
# Optional: --data-path /absolute/path/to/your.csv
```

### Models Overview
- Realistic: Uses only pre-flight features (time/route/airports/distance). Includes probability calibration (isotonic), thresholding targeting recall≈0.80, and a delay-bucket classifier (0, 1–15, 16–60, >60) as the primary output. Regression for exact minutes is kept but de‑prioritized.
- Consumer: Two-stage (classification → regression) with calibration, F1-aware thresholding, and optional boosters (LightGBM/XGBoost/CatBoost) if installed.
- Operational: Focused on actionable drivers (e.g., LateAircraft/Carrier/NAS/Weather binaries) for diagnosis and planning; strong bucket performance on time-based splits.

## Repro Tips
- Use time-based splits for out-of-time testing (e.g., 2018–2019 → 2022–2023) when full years exist.
- Class imbalance is handled via `class_weight='balanced'` and threshold tuning.
- Feature encodings are label-based for categorical variables.

## Quick Recommendations
- Pre-departure prediction: Avoid leakage features like `WeatherDelay`, `CarrierDelay`, `NASDelay`, `LateAircraftDelay` in inputs. Prefer rolling/expanding route/airport priors and traffic proxies computed from training history only.
- Optimize the decision rule: Calibrate probabilities (isotonic) and pick a threshold for your target (e.g., recall=0.80) on a hold-out period.
- Favor bucket classification over minute regression for operational decisions; report macro-F1 and per-class F1.
- Try simple boosting (LightGBM/XGBoost/CatBoost) with minimal tuning; report ROC-AUC and PR-AUC.
- Add external signals when available (weather/ATC/holidays) for further gains.

## Decisions & Rationale (What and Why)

### Preprocessing
- Data quality checks: Early missing/duplicate scans to surface issues before feature work and prevent silent bias.
- Redundant columns removal: Avoid duplicate signals (e.g., `DepDelay` vs `DepDelayMinutes`, display variants of time fields) to reduce leakage and multicollinearity.
- Typed time features: Convert `FlightDate` → `DepDate` and derive `Year`, `Month`, `Day`, `DayOfWeek`, `Quarter`, `Season` to capture calendar/seasonality effects.
- Route construction: `Route = OriginCity → DestCity` exposes airport-pair effects critical for delay patterns.
- COVID filtering (2020–2021): Shifts in behavior make those years unrepresentative for general models; removed for more stable training and OOT testing.
- DelaySeverity: Quick interpretability for downstream analysis/plots and bucket modeling.

### Data Analysis
- Route analysis: Identify high-delay routes for operational prioritization.
- Seasonal/temporal patterns: Day-of-week/hour-of-day effects influence staffing/scheduling decisions.
- STL decomposition: Separates trend/seasonal/residual components to understand systematic vs random structure.
- Cause breakdown (where available): Weather/Carrier/NAS/LateAircraft help distinguish controllable vs external drivers.

### Machine Learning
- Leakage policy: “Realistic” model excludes contemporaneous operational delay fields; only pre-departure features (time/route/airports/distance). Operational pipeline may include cause binaries for diagnosis, not for pre-departure prediction.
- Targets: Primary target shifted to delay buckets (0, 1–15, 16–60, >60) for actionable decisions; minute regression retained but de‑prioritized due to low R².
- Class imbalance: `class_weight='balanced'` and threshold tuning to align with operational preferences.
- Calibration & thresholding: Isotonic calibration improves probability quality; thresholds are selected for a target recall (e.g., 0.80) on a hold-out period to control miss rate.
- Model selection: Quick comparison among RandomForest and boosting models (LightGBM/XGBoost/CatBoost), choosing best by PR-AUC on validation.
- Leakage-safe priors (where implemented): Route/airport rolling or expanding aggregates computed on training history only to avoid future information.

### Validation & Evaluation
- Time-based splits where full-year data permits (e.g., 2018–2019 → 2022–2023) to evaluate generalization across periods.
- Report ROC-AUC, PR-AUC for classification; macro-F1 for buckets; MAE and R² for regression (informational only).
- Feature importance (tree-based) for interpretability; aligns with observed temporal/route effects.

### Practical Recommendations
- For pre-departure prediction: avoid leakage features (`WeatherDelay`, `CarrierDelay`, `NASDelay`, `LateAircraftDelay`). Prefer rolling priors and traffic proxies from training-only history.
- Optimize decisions: calibrate probabilities and pick a threshold for your target (e.g., recall=0.80) on a clean hold-out.
- Prefer bucket classification for operations; track macro-F1 and class-F1. Add simple boosting with minimal tuning and report ROC/PR-AUC.
- Enrich with external signals (weather/ATC/holidays) when available to improve predictability.