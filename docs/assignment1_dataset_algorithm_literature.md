## Assignment 1 — Dataset & Algorithm Selection and Literature Review

### 1) Introduction

This project forecasts monthly retail turnover at the intersection of state and industry. The goal is to build predictive models that provide accurate short‑horizon forecasts to support planning and policy analysis. The work uses an open secondary dataset of monthly retail turnover and implements two deep learning approaches selected for their complementary strengths in tabular and sequential modeling.

---

### 2) Dataset & Algorithm Selection

#### Problem definition

- **Task**: Supervised regression — predict monthly retail turnover for each series.
- **Target**: `Turnover` (numeric); **horizon**: one month ahead (H=1).
- **Granularity**: Series identified by `State × Industry` (`Series ID`), timestamped by monthly `Month`.

#### Dataset description and source

- Dataset: `aus_retail` from the R `tsibbledata` package, exported to `notebooks/data/retail.csv` for this project.
- Time period: monthly data from 1982-04 to 2018-12 (1982–2018).
- Variables: `State`, `Industry`, `Series ID` (unique state–industry identifier), `Month` (year–month), and `Turnover` (millions of Australian dollars).
- Size: 64,532 rows and 5 columns in this snapshot.
- Source and license: Australian Bureau of Statistics “Retail Trade, Australia” series, distributed via `tsibbledata`. Cite both ABS and `tsibbledata` documentation in the report.

Dataset overview: The Australian Retail Trade dataset (ABS), distributed as `aus_retail` via the `tsibbledata` R package, is a high‑quality multi‑series time‑series resource for economic forecasting. It records monthly retail turnover, measured in millions of Australian dollars, by state/territory and industry, enabling analysis of trends and seasonality across the national retail sector. The dataset used in this study is the full state–industry panel spanning April 1982 to December 2018 and comprising 64,532 monthly observations. Each record includes `State`, `Industry`, `Series ID`, `Month`, and `Turnover`. For this project, the tsibble was exported to CSV for reproducible model training and evaluation.

#### Suitability and critical analysis

- **Size/fit**: Medium–large multi‑series dataset appropriate for deep learning when augmented with simple covariates.
- **Temporal structure**: Clear trend and strong annual seasonality; per‑series histories are long enough for sequence models.
- **Quality checks**: Duplicates removed; core fields coerced to correct types; negative/zero turnovers removed where inappropriate for percentage metrics. Missing values in `Month`/`Turnover` dropped. Summary EDA includes distributions, time trends, and cross‑sectional boxplots (see `notebooks/01_eda_data_prep.ipynb`).
- **Risks/limitations**: Structural breaks across decades; unequal series lengths; potential revisions to official statistics. MAPE is unstable near zero — reported alongside RMSE/MAE.

#### Prediction protocol

- **Split**: Chronological split by unique months (≈70% train, 15% validation, 15% test) to avoid leakage.
- **Metrics**: MSE/RMSE (scale‑sensitive), MAE (robust), and MAPE (interpretable but used with caution).

#### Selected models and rationale

1. **Deep Neural Network (DNN, tabular baseline)**

   - Inputs: one‑hot encodings of `State`, `Industry`, `Series ID` + calendar features (`Year`, `MonthNum`, `Quarter`, sine/cosine of month).
   - Why: Strong cross‑sectional signal from categorical hierarchies; simple, fast baseline; well‑understood regularization (batch norm, dropout) and training with Adam on MSE.

2. **GRU sequence model (Seq2Seq)**
   - Inputs: scaled recent history (`TurnoverScaled`) plus cyclical time features and a normalized series index (`SeriesIndexNorm`). Uses a rolling **window = 12 months** to predict **horizon = 1**.
   - Why: Gated RNNs (GRU/LSTM) capture temporal dependencies and seasonality more effectively than feed‑forward models, typically yielding lower forecast error on long monthly series.

The pair offers complementary inductive biases: DNN exploits cross‑sectional heterogeneity; GRU focuses on temporal dynamics. Model selection is based on validation RMSE/MAE, stability across seeds, and computational efficiency. Hyperparameters are tuned with Bayesian search to produce the final models.

---

### 3) Literature Review (focused on retail time‑series forecasting)

#### Classical statistical baselines

- Exponential smoothing and ARIMA remain strong for many seasonal series and provide interpretable components and principled uncertainty. Hierarchical reconciliation improves coherence across `State`/`Industry` groupings.

#### Deep learning for time series

- **Feed‑forward DNNs** on engineered time/categorical features can perform competitively when cross‑sectional information is rich but temporal dependencies are short.
- **Recurrent architectures (LSTM/GRU)** model long‑range temporal dependencies via gating and often outperform classical methods on large multi‑series datasets with complex seasonality.
- **Temporal CNNs and Transformers** (e.g., TCN, N‑BEATS, TFT) represent additional state‑of‑the‑art approaches. They can provide gains with sufficient data/compute but add complexity beyond the scope and constraints of this assignment; hence they are referenced but not selected.

#### Feature representation and preprocessing

- Cyclical time encodings (sine/cosine) avoid artificial discontinuities at year boundaries; scaling targets stabilizes sequence training; one‑hot encodings represent categorical hierarchies. Care is taken to avoid information leakage by constructing features from past or contemporaneous information only.

#### Hyperparameter tuning and regularization

- Early stopping, dropout, batch/layer normalization, and learning‑rate scheduling are standard. Bayesian hyperparameter optimization (via Keras Tuner) efficiently searches architectures and learning rates under compute limits.

#### Evaluation methodology

- Time‑ordered validation and a held‑out test set are essential. RMSE/MAE are robust scale metrics; MAPE is complementary but unreliable for small denominators, so it is interpreted cautiously.

#### Synthesis and justification

- Prior work consistently finds gated RNNs (GRU/LSTM) strong on monthly retail and similar demand problems, especially with long histories and seasonality, while feed‑forward baselines remain competitive when cross‑sectional signals dominate. This motivates selecting a GRU as the primary sequential model and a DNN as a tabular baseline. The chosen metrics and split strategy align with best practices for temporal generalization.

#### References (illustrative)

1. Hyndman, R.J., & Athanasopoulos, G. Forecasting: Principles and Practice (3e). Monash University. Available online.
2. Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. Time Series Analysis: Forecasting and Control.
3. Taylor, S.J., & Letham, B. Forecasting at Scale (Prophet). PeerJ Preprints.
4. Hyndman, R.J., Ahmed, R.A., Athanasopoulos, G., & Shang, H.L. Optimal combination forecasts for hierarchical time series. Computational Statistics & Data Analysis.
5. Hochreiter, S., & Schmidhuber, J. Long Short‑Term Memory. Neural Computation.
6. Cho, K. et al. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation (GRU).
7. Salinas, D., Flunkert, V., Gasthaus, J., & Tim Januschowski. DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks.
8. Bai, S., Kolter, J.Z., & Koltun, V. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling (TCN).
9. Oreshkin, B.N., Carpov, D., Chapados, N., & Bengio, Y. N‑BEATS: Neural basis expansion analysis for interpretable time series forecasting.
10. Lim, B., Arik, S.Ö., et al. Temporal Fusion Transformers for Interpretable Multi‑horizon Time Series Forecasting.
11. O’Malley, T. et al. Keras Tuner: A Scalable Hyperparameter Tuning Library for Keras.
12. Australian Bureau of Statistics. Retail Trade, Australia — official statistics portal and metadata.

Note: Replace or augment the above with the exact sources you choose to cite, adding URLs and access dates per your referencing style.

---

### 4) Data Preparation and EDA

This section documents the end‑to‑end preparation applied to `notebooks/data/retail.csv`, with explicit references to notebook cells for easy screenshotting and traceability.

Overview

- Core EDA and cleaning are in `notebooks/01_eda_data_prep.ipynb`.
- Model‑specific preprocessing (feature scaling/encoding and sequence windowing) occurs in `notebooks/02_model_building.ipynb` and `notebooks/03_model_tuning.ipynb`.

Steps and notebook cell references

1. Load and initial profile (Cell 2–3, `01_eda_data_prep.ipynb`)

   - Load CSV, print shape, preview `head()`.
   - Build a quick profile dictionary: column names, dtypes, row/column counts, and missing‑value percentages.

2. Type parsing and coercion (Cell 4, `01_eda_data_prep.ipynb`)

   - Parse `Month` with format `YYYY Mon` to datetime.
   - Clean string quotes if present; cast `State`, `Industry`, `Series ID` to categorical.
   - Coerce `Turnover` to numeric.

3. Quality checks (Cell 5, `01_eda_data_prep.ipynb`)

   - Report missing counts per column and total duplicate rows.

4. Exploratory visuals (Cell 6, `01_eda_data_prep.ipynb`)

   - Distribution of `Turnover`, boxplots by `State`, and overall time trend.

5. Cleaning rules (Cell 7, `01_eda_data_prep.ipynb`)

   - Drop exact duplicates.
   - Drop rows missing `Month` or `Turnover`.
   - Remove non‑positive `Turnover` values (basic domain sanity).

6. Post‑clean EDA (Cell 8, `01_eda_data_prep.ipynb`)

   - Repeat key plots on the cleaned frame: distribution, time trend, boxplots.

7. Feature engineering — global (Cell 9, `01_eda_data_prep.ipynb`)

   - Derive `Year`, `MonthNum`, `Quarter`, `YearMonth`.
   - Example aggregation: state‑industry monthly totals.

8. Preprocessing scaffold (Cell 10, `01_eda_data_prep.ipynb`)

   - `ColumnTransformer` skeleton combining `StandardScaler` for numeric and `OneHotEncoder` for categorical features.

9. Time‑based data split and artifact saves (Cell 11, `01_eda_data_prep.ipynb`)
   - Chronological split by unique months: 70% train, 15% validation, 15% test.
   - Outputs saved to `notebooks/data/processed/{train,val,test}.csv`.

Model‑specific preprocessing (for completeness in this section)

- Cyclical time features and series index (Cell 6, `02_model_building.ipynb`)

  - Add `MonthSin` and `MonthCos` from `MonthNum`.
  - Map `Series ID` to `SeriesIndex` and normalized `SeriesIndexNorm`.
  - Scale `Turnover` and `Year` with `StandardScaler` (temporary in Task 1).

- Tabular DNN preprocessing (Cell 8, `02_model_building.ipynb`; Cell 5, `03_model_tuning.ipynb`)

  - One‑hot encode `State`, `Industry`, `Series ID`; standardize `Year`, `MonthNum`, `Quarter`, `MonthSin`, `MonthCos` using a `ColumnTransformer`.
  - Final preprocessor persisted in tuning as `artifacts/dnn_preprocessor.joblib` (Cell 5, `03_model_tuning.ipynb`).

- GRU sequence preparation (Cell 14, `02_model_building.ipynb`; Cells 4 & 6, `03_model_tuning.ipynb`)
  - Persist `StandardScaler` objects for `Turnover` and `Year` as `artifacts/turnover_scaler.joblib` and `artifacts/year_scaler.joblib` (Cell 4, `03_model_tuning.ipynb`).
  - Build rolling windows of length 12 with horizon 1 from features: `TurnoverScaled`, `MonthSin`, `MonthCos`, `YearScaled`, `SeriesIndexNorm` (Cell 14 in `02_model_building.ipynb`; Cell 6 in `03_model_tuning.ipynb`).

Artifacts and paths (for reproducibility screenshots)

- Processed splits (Cell 11, `01_eda_data_prep.ipynb`): `notebooks/data/processed/{train,val,test}.csv`.
- Scalers (Cell 4, `03_model_tuning.ipynb`): `notebooks/artifacts/turnover_scaler.joblib`, `notebooks/artifacts/year_scaler.joblib`.
- DNN preprocessor (Cell 5, `03_model_tuning.ipynb`): `notebooks/artifacts/dnn_preprocessor.joblib`.

Notes for the report

- When capturing screenshots, include the cell number and the printed outputs/figures listed above.
- Use the pre‑clean vs post‑clean plots (Cells 6 and 8 in `01_eda_data_prep.ipynb`) to illustrate impact of cleaning.
- Justify the chronological split to avoid information leakage common in time‑series problems.

---

### 5) Model Building

This section implements two complementary deep learning models using the processed splits from Section 4:

- Deep Neural Network (tabular baseline)
- GRU sequence model (windowed time series)

All steps are in `notebooks/02_model_building.ipynb`. Cell references below indicate what to screenshot and explain in the report.

Core workflow and cells to explain

1. Load processed datasets (Cell 4)

   - Loads `train.csv`, `val.csv`, `test.csv` saved in Section 4 and prints shapes. This confirms reproducible inputs for both models.

2. Shared feature engineering and scaling (Cell 6)

   - Adds cyclical encodings `MonthSin`/`MonthCos` and constructs `SeriesIndex`/`SeriesIndexNorm` from `Series ID`.
   - Standardizes `Turnover` and `Year` (used by the GRU pipeline). Explain why scaling stabilizes training.

3. Tabular DNN preprocessing (Cell 8)

   - `ColumnTransformer` one‑hot encodes `State`, `Industry`, `Series ID` and scales numeric calendar features (`Year`, `MonthNum`, `Quarter`, `MonthSin`, `MonthCos`).
   - Shows the resulting feature dimension (e.g., 183). This is the exact tabular input to the DNN.

4. Tabular DNN model, training, and evaluation (Cells 10–12)

   - Cell 10: Define a 3‑hidden‑layer dense network with batch norm and dropout; compile with Adam on MSE and track RMSE/MAE/MAPE. Train with early stopping and LR reduction.
   - Cell 11: Evaluate on the test set and compute additional metrics (RMSE, MAE, MAPE, R²). Screenshot the printed metrics.
   - Cell 12: Plot training/validation loss curves. Optional screenshot to evidence convergence.

5. GRU sequence preparation, model, and evaluation (Cells 14–18)

   - Cell 14: Build rolling windows (window = 12, horizon = 1) using features [`TurnoverScaled`, `MonthSin`, `MonthCos`, `YearScaled`, `SeriesIndexNorm`]. Screenshot the sequence shapes printed.
   - Cell 15: Create efficient `tf.data` pipelines for training/validation/testing.
   - Cell 16: Define a two‑layer GRU network (128 → 64 with normalization and dropout) and train with early stopping/LR reduction.
   - Cell 17: Inverse‑transform predictions back to the original turnover scale and compute metrics (RMSE/MAE/MAPE/R²). Screenshot the printed metrics.
   - Cell 18: Plot GRU training/validation loss curves. Optional screenshot for convergence.

6. Model comparison (Cell 20)
   - Assemble a small table comparing DNN vs GRU on key metrics. Useful for a brief narrative on which inductive bias performs better and why.

Training details to mention in prose

- Optimizer: Adam; Loss: MSE; Metrics: MSE, RMSE, MAE, MAPE.
- Regularization: batch/layer normalization and dropout; early stopping with best‑weights restore; ReduceLROnPlateau.
- Reproducibility: fixed random seed (42) and deterministic preprocessing from the processed CSVs.

Minimum screenshots for the report (keep concise)

- Cell 4 (dataset shapes), Cell 6 (shared features/scaling), Cell 8 (DNN preprocessor output dim), Cell 10 (DNN model summary/training call), Cell 11 (DNN test metrics), Cell 14 (sequence shapes), Cell 16 (GRU model/training call), Cell 17 (GRU test metrics), Cell 20 (comparison table).

#### 5.1 Deep Neural Network (DNN) — Tabular Baseline

Objective

- Learn cross‑sectional patterns across `State`, `Industry`, and `Series ID`, combined with calendar features, to predict monthly turnover directly in tabular form.

Inputs and preprocessing

- Categorical: `State`, `Industry`, `Series ID` — one‑hot encoded.
- Numeric: `Year`, `MonthNum`, `Quarter`, `MonthSin`, `MonthCos` — standardized.
- Implementation: `ColumnTransformer` in Cell 8 creates the final tabular matrix; feature dimension printed (e.g., 183) is the exact input to the network.

Architecture and training (Cells 10–12)

- Network: 3 hidden dense layers with ReLU, batch normalization, and dropout; output is a single linear neuron for turnover.
- Loss/metrics: MSE optimized by Adam; track RMSE/MAE/MAPE for interpretability.
- Regularization: early stopping (best weights) and ReduceLROnPlateau.
- Evidence: screenshot training call (Cell 10), test metrics (Cell 11), and loss curve (Cell 12).

Interpretation

- The DNN captures differences across states/industries and seasonal calendar signals without explicit temporal recurrence. Performance reflects the strength of cross‑sectional covariates and global seasonality proxies (sin/cos), serving as a strong and fast baseline.

Screenshots to include

- Cell 8 (preprocessor output shape)
- Cell 10 (model summary or compile/fit call)
- Cell 11 (printed test metrics)
- Cell 12 (training/validation loss plot)

#### 5.2 Recurrent Neural Network — GRU Sequence Model

Objective

- Model temporal dynamics by feeding recent monthly windows to a GRU to predict next‑month turnover.

Inputs and sequence windowing

- Feature window per series: [`TurnoverScaled`, `MonthSin`, `MonthCos`, `YearScaled`, `SeriesIndexNorm`].
- Window/horizon: 12 → 1.
- Implementation: Cell 14 builds arrays per series ordered by `Month` and prints sequence shapes; Cell 15 builds efficient `tf.data` pipelines.

Architecture and training (Cells 16–18)

- Network: two GRU layers (128 then 64), layer normalization and dropout; dense head outputs the scaled target.
- Loss/metrics: MSE with Adam; monitor RMSE/MAE/MAPE.
- Postprocessing: inverse‑transform predictions from scaled back to original turnover (Cell 17) before computing metrics.
- Evidence: screenshot training call (Cell 16), printed metrics after inverse transform (Cell 17), and loss curves (Cell 18).

Interpretation

- The GRU leverages temporal dependencies and seasonal structure present in each series’ recent history, typically outperforming tabular baselines when sequential patterns dominate. Using cyclical encodings and a normalized series index helps the model share seasonal knowledge and series identity cues.

Screenshots to include

- Cell 14 (sequence shapes)
- Cell 16 (model summary or compile/fit call)
- Cell 17 (printed test metrics after inverse transform)
- Cell 18 (training/validation loss plot)
