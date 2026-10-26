"""Reproduce the Project 1 competition submission end to end.

The submitted model is a five-seed probability ensemble of the 16-unit MLP
selected by the ablation in ``experiments/nn_ablation.py``.  Running

    python run.py

from the repository root performs the following steps:

1. load ``dataset/x_train.csv``, ``dataset/y_train.csv`` and ``dataset/x_test.csv``;
2. build the design matrix (mean imputation, per-column missing indicators,
   dropping constant/all-missing columns, standardisation), fitting every
   statistic on training rows only;
3. split the labelled data into stratified train / validation / holdout
   (64 / 16 / 20), train the MLP with seeds 42-46 on the 64% training split and
   average the five predicted probabilities;
4. calibrate the decision threshold on the 16% validation split;
5. retrain the same five-seed ensemble on all labelled rows;
6. average the five test probabilities and write ``results/y_test.csv`` in the
   ``Id,Prediction`` format.

The output file is bit-for-bit identical to the CSV submitted to the
competition system (competition F1 = 0.444, accuracy = 0.882).  Training takes
about six minutes on a laptop CPU; the per-iteration optimiser and network code
are in ``implementations.py``, and the preprocessing/metrics code is in
``nn_pipeline.py``.  The controlled ablation, the same-split comparison of all
course methods and the holdout bootstrap are in ``experiments/``.

Only NumPy and the standard library are used for the model (matplotlib is used
by the reporting helpers); no external ML library or external dataset is used.
"""

from pathlib import Path

import numpy as np

import nn_pipeline as nnp
from implementations import neural_network_predict, neural_network_train

SEED = 42
# Selected by experiments/nn_ablation.py: one hidden layer of 16 ReLU units,
# positive-class weight 3 and L2 weight decay 1e-3.
BEST_CONFIG = dict(hidden_dims=(16,), pos_weight=3.0, l2=1e-3)
TRAIN_KWARGS = dict(gamma=1e-3, batch_size=1024, optimizer="adam", max_iters=8000)
ENSEMBLE_SEEDS = (42, 43, 44, 45, 46)


def load_training_matrix(root, raw):
    """Return the cached float32 training matrix, without the ``Id`` column.

    The cache is shared with the ablation scripts: it makes the training matrix
    identical across experiments and avoids re-parsing the 217 MB CSV.  It is
    rebuilt automatically on first use (about one minute).
    """
    cache = root / ".cache"
    cache.mkdir(exist_ok=True)
    x_path, y_path = cache / "x_train.npy", cache / "y_train.npy"
    if not (x_path.exists() and y_path.exists()):
        np.save(x_path, raw["x_train"].astype(np.float32))
        np.save(
            y_path,
            np.column_stack([raw["train_ids"], raw["y_train"]]).astype(np.float32),
        )
    x = np.load(x_path)
    if x.shape[1] == 322:  # legacy cache built with the Id column
        x = np.ascontiguousarray(x[:, 1:])
        np.save(x_path, x)
    return x


def ensemble_predictions(z_fit, y_fit, z_eval):
    """Train the five seeds and average their positive-class probabilities."""
    probabilities = []
    for seed in ENSEMBLE_SEEDS:
        params, _ = neural_network_train(
            (y_fit == 1).astype(float),
            z_fit,
            seed=seed,
            **BEST_CONFIG,
            **TRAIN_KWARGS,
        )
        probabilities.append(neural_network_predict(z_eval, params))
    return np.mean(probabilities, axis=0)


def main():
    """Train the ensemble, calibrate the threshold and write the submission."""
    root = Path(__file__).resolve().parent
    raw = nnp.load_raw(root / "dataset")
    x_train = load_training_matrix(root, raw)
    y = raw["y_train"]

    # --- Stratified 64/16/20 split (identical to the notebook/ablation) -----
    dev_idx, _ = nnp.stratified_split(y, val_fraction=0.20, seed=SEED)
    train_local, val_local = nnp.stratified_split(
        y[dev_idx], val_fraction=0.20, seed=SEED + 1
    )
    train_idx, val_idx = dev_idx[train_local], dev_idx[val_local]

    z_train, info = nnp.fit_design_matrix(x_train[train_idx], add_indicators=True)
    z_val = nnp.transform_design_matrix(x_train[val_idx], info)

    # --- Calibrate the decision threshold on validation ---------------------
    val_probs = ensemble_predictions(z_train, y[train_idx], z_val)
    threshold, val_f1 = nnp.best_f1_threshold(
        (y[val_idx] == 1).astype(float), val_probs
    )
    print(f"Validation ensemble F1 = {val_f1:.4f} at threshold {threshold:.4f}")

    # --- Refit on all labelled rows and predict the test set ----------------
    z_all, info_all = nnp.fit_design_matrix(x_train, add_indicators=True)
    z_test = nnp.transform_design_matrix(raw["x_test"], info_all)
    test_probs = ensemble_predictions(z_all, y, z_test)

    submission = nnp.save_submission(
        raw["test_ids"],
        test_probs,
        root / "results" / "y_test.csv",
        threshold=threshold,
    )
    predicted = int((test_probs >= threshold).sum())
    print(f"Wrote {submission} with {len(test_probs)} rows ({predicted} positive).")


if __name__ == "__main__":
    main()
