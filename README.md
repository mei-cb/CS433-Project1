# CS-433 Project 1 — CHD classification

Binary classification of `_MICHD` (myocardial infarction / coronary heart disease)
on the `dataset/` split of the EPFL Machine Learning Project 1. Labels are `-1`
(no CHD) and `+1` (CHD); only ~8.8% of the rows are positive and ~44.8% of the
feature cells are missing.

The final competition submission is a **five-seed probability ensemble** of a
16-unit MLP. It achieved **F1 = 0.444 and accuracy = 0.882** on the competition
test set. `run.py` reproduces the submitted `results/y_test.csv`
**bit-for-bit** (109,379 rows, 13,608 positive predictions, validation threshold
0.4548).

Only the Python standard library, **NumPy** and **matplotlib** are used — no
pandas, scikit-learn or deep-learning framework, and no external datasets. Every
model, the preprocessing, the metrics (ROC-AUC, average precision,
precision/recall/F1) and the CSV I/O are implemented from scratch.

## Repository layout

| Path | Purpose |
| --- | --- |
| `run.py` | Reproduces the competition submission (`results/y_test.csv`) end to end: 5-seed ensemble, validation threshold, refit on all labelled rows. |
| `implementations.py` | Six required course methods plus the from-scratch neural-network trainer (`neural_network_train`, `neural_network_predict`). |
| `nn_pipeline.py` | Data loading, preprocessing (imputation, missing indicators, standardisation), metrics and plotting helpers. |
| `notebooks/03_neural_network.ipynb` | Documented neural-network pipeline and single-seed demonstration; writes local result files only. |
| `grading_tests/` | Public course tests for the six basic methods. |
| `setup_test.ipynb`, `helpers.py` | Course scaffolding and test helpers. |
| `requirements.txt` | `numpy` and `matplotlib`. |
| `results/y_test.csv` | The submitted predictions (`Id,Prediction`). |

The two-page PDF report, the ablation scripts and the additional result figures
are **not** part of this code-only repository; they are kept locally. Their
findings are summarised below.

## Data preparation and feature generation

The raw training matrix has 321 features and 328,135 labelled rows; the `Id`
column is used only for the submission format. The preprocessing, implemented in
`nn_pipeline.fit_design_matrix`, is:

1. drop constant and all-missing columns (6 columns);
2. mean-impute the 315 remaining columns with their observed training mean;
3. standardise each column with its observed training mean and standard
   deviation;
4. append one binary **missing-value indicator** for each of the 236 columns
   that contain missing values (informative missingness);
5. append a 1-column intercept for the linear course methods only.

The result is a 551-column neural-network design matrix (552 including the
intercept for the linear baselines). Every statistic is estimated on training
rows only and applied unchanged to validation, holdout and test rows, so there is
no leakage. The test CSV is transformed with statistics fitted on all labelled
rows for the final submission.

## Validation / cross-validation protocol

We use a single **stratified 64/16/20 split** rather than k-fold
cross-validation:

* 20% holdout (65,627 rows) — internal test set, never used for fitting, model
  selection or threshold tuning;
* the remaining 80% dev set is split into 64% train and 16% validation;
* the validation split selects the architecture/regularisation and the
  F1-optimal decision threshold;
* the selected configuration is refit on the full dev set and evaluated once on
  the holdout for the numbers below;
* the competition model is refit on all labelled rows and uses the validation
  threshold.

Five seeds (42–46) capture initialisation variance; split variance is not
captured by this protocol. The controlled five-seed ablation is summarised in the
report.

## Model

The submitted model is a fully connected MLP with one hidden ReLU layer
(`551 → 16 → 1`) and a sigmoid output. It minimises class-weighted binary
cross-entropy with L2 weight decay:

```
L = -1/N Σ a_i [y_i log p_i + (1-y_i) log(1-p_i)] + λ ||W||²
a_i = 3 if y_i = 1 else 1
```

with `λ = 1e-3`, Adam (`lr = 1e-3`), mini-batches of 1024 and 8000 iterations.
He initialisation and back-propagation are implemented from scratch in
`implementations.py`. At prediction time the probabilities of seeds 42–46 are
averaged; the validation threshold is 0.4548.

## Implemented course methods

`implementations.py` contains the six required methods, each with docstrings:

1. `mean_squared_error_gd` — MSE linear regression with gradient descent;
2. `mean_squared_error_sgd` — MSE linear regression with stochastic gradient descent;
3. `least_squares` — closed-form least squares (normal equations);
4. `ridge_regression` — closed-form ridge regression;
5. `logistic_regression` — logistic regression with gradient descent;
6. `reg_logistic_regression` — L2-regularised logistic regression with gradient descent.

The notebook and report describe the neural-network trainer built on top of the
same NumPy utilities.

## Results

| Model | Split | F1 | Precision | Recall | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- | --- |
| MLP-16, 5-seed ensemble (**submitted**) | holdout | **0.4350** | 0.365 | 0.538 | **0.8653** | **0.4037** |
| MLP-16, single seed (reference) | holdout | 0.4342 | 0.340 | 0.602 | 0.8629 | 0.3961 |
| Ridge (λ = 1e-3) | holdout | 0.4233 | 0.342 | 0.556 | 0.8507 | 0.3956 |
| MLP-16, 5-seed mean | validation | 0.4311 | 0.359 | 0.541 | 0.8601 | 0.3906 |
| Ridge (λ = 1e-3) | validation | 0.4222 | 0.342 | 0.551 | 0.8500 | 0.3935 |

**Competition test:** F1 = **0.444**, accuracy = **0.882** (test labels are not
available locally). Paired bootstrap on the internal holdout: MLP ensemble vs.
ridge gives +0.015 ROC-AUC (95% CI 0.012–0.017), +0.012 F1 (0.005–0.018) and
+0.008 AP (0.001–0.016); all three intervals exclude zero.

## Key findings (five-seed ablation)

* The **ReLU nonlinearity** and **L2 weight decay** are the largest contributors:
  removing either costs 0.010–0.011 validation F1.
* **Class weighting is neutral after threshold calibration.** It raises F1 at the
  default 0.5 threshold from 0.198 to 0.425, but tuned F1 is essentially
  unchanged (0.434 vs 0.431) and AP is slightly higher without weighting
  (0.393 vs 0.391). The imbalance is better handled at decision time.
* **Per-column missing indicators** help modestly (removing all 236 costs 0.006
  F1 and 0.013 AP); a single missing-count feature does not recover the loss.
* **Depth does not help**: 64 units change F1 by −0.003 and a 64–32 network by
  −0.008.
* **Seed ensembling** adds about 0.007 validation F1 and 0.003 AUC; it is the
  submitted model.

## Reproduce

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py            # writes the submitted results/y_test.csv
```

`run.py` takes about six minutes on a laptop CPU. It prints the validation F1
and threshold (0.4379 / 0.4548) and writes 109,379 rows with 13,608 positive
predictions. The output is bit-for-bit identical to the submitted file
(SHA-256 `ed2160c2...`). The training code is entirely in `implementations.py` /
`nn_pipeline.py`; there are no pretrained weights to download.

The dataset is provided by the course and is not committed. Place
`x_train.csv`, `y_train.csv` and `x_test.csv` in `dataset/` before running.
The raw feature cache (`.cache/`) is rebuilt automatically on first use.

To run the documented notebook, install `nbformat`, `nbconvert`,
`jupyter_client` and `ipykernel`, then open
`notebooks/03_neural_network.ipynb` (kernel `project1-nn`) or execute it with
`jupyter nbconvert --execute`.
