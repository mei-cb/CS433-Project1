"""Reusable utilities for the Project 1 neural-network pipeline.

Only the Python standard library and NumPy are used for the data handling,
metrics and submission writing; matplotlib (a plotting library) is used to save
the vector figures for the report. No pandas / scikit-learn / deep-learning
framework is imported anywhere.

The module groups the three things the neural-network notebook needs:

* **Data loading and preprocessing** - ``load_raw`` and the design-matrix
  builders perform per-column mean imputation, drop constant / all-missing
  columns, append missing-value indicators and standardise the features. All
  statistics are estimated on the training rows only, so the validation and
  test transformations never leak information from the evaluation data.
* **Evaluation metrics** - confusion-matrix based metrics (precision, recall,
  F1, accuracy), threshold-free ranking metrics (ROC-AUC, average precision)
  and an F1-optimal threshold search, all implemented with NumPy.
* **Plotting and reporting** - helpers that save vector (PDF) figures and write
  the CSV tables used in the report.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are written to files.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Use a compact, report-friendly style.
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

ID_COLUMN = "Id"
LABEL_COLUMN = "_MICHD"


def project_root() -> Path:
    """Return the repository root (the folder that contains ``dataset/``)."""
    for path in (Path.cwd(), *Path.cwd().parents):
        if (path / "dataset").is_dir() and (path / "implementations.py").is_file():
            return path
    raise FileNotFoundError("Open the notebook from the project or notebooks folder.")


def _read_header(path) -> list:
    """Return the column names of a CSV file (standard-library csv reader)."""
    with open(path, newline="") as handle:
        return next(csv.reader(handle))


def load_raw(data_dir) -> dict:
    """Load the raw CSVs and return ids, features and signed labels.

    Uses ``numpy.genfromtxt``, which parses empty fields as ``nan``; the
    preprocessing step later imputes them.

    Returns:
        dict with keys ``x_train``, ``y_train``, ``train_ids``, ``x_test``,
        ``test_ids`` and ``feature_names``. Labels keep the ``{-1, +1}``
        encoding of the data.
    """
    data_dir = Path(data_dir)

    header = _read_header(data_dir / "x_train.csv")
    feature_names = np.asarray([c for c in header if c != ID_COLUMN])

    train = np.genfromtxt(
        data_dir / "x_train.csv", delimiter=",", skip_header=1, dtype=np.float64
    )
    train_ids = train[:, 0]
    x_train = train[:, 1:]

    test = np.genfromtxt(
        data_dir / "x_test.csv", delimiter=",", skip_header=1, dtype=np.float64
    )
    test_ids = test[:, 0]
    x_test = test[:, 1:]

    labels = np.genfromtxt(
        data_dir / "y_train.csv", delimiter=",", skip_header=1, dtype=np.float64
    )
    y_train = labels[:, 1]

    return {
        "x_train": x_train,
        "y_train": y_train,
        "train_ids": train_ids,
        "x_test": x_test,
        "test_ids": test_ids,
        "feature_names": feature_names,
    }


def _apply_design_matrix(X, info, add_indicators):
    """Transform ``X`` using statistics estimated by ``fit_design_matrix``."""
    X = np.asarray(X, dtype=np.float64)
    nan_mask = np.isnan(X)
    means = info["means"]
    stds = info["stds"]
    keep = info["keep"]

    # Mean-impute and standardise the kept columns only (saves memory).
    filled = X[:, keep].copy()
    sub_nan = nan_mask[:, keep]
    filled[sub_nan] = np.broadcast_to(means[keep], filled.shape)[sub_nan]
    Z = (filled - means[keep]) / stds[keep]

    if add_indicators:
        indicators = nan_mask[:, info["indicator_mask"]].astype(np.float32)
        Z = np.column_stack([Z, indicators])

    return np.ascontiguousarray(Z, dtype=np.float32)


def fit_design_matrix(X_train, add_indicators=True, min_std=1e-8):
    """Fit imputation/scaling statistics on ``X_train`` and transform it.

    Missing entries are replaced by the observed column mean, constant and
    all-missing columns are dropped, remaining columns are standardised and a
    binary indicator is appended for every column that has missing values.

    Args:
        X_train: Raw training features, shape (N, D).
        add_indicators: Append missing-value indicators.
        min_std: Columns with a smaller standard deviation are dropped.

    Returns:
        Z: Transformed training design matrix (float32).
        info: dict of fitted statistics for :func:`transform_design_matrix`.
    """
    X = np.asarray(X_train, dtype=np.float64)
    nan_mask = np.isnan(X)
    observed = ~nan_mask
    counts = observed.sum(axis=0)

    sums = np.where(observed, X, 0.0).sum(axis=0)
    means = np.divide(
        sums, counts, out=np.zeros(X.shape[1], dtype=np.float64), where=counts > 0
    )
    centered = np.where(observed, X - means, 0.0)
    variances = (centered**2).sum(axis=0) / np.maximum(counts, 1)
    stds = np.sqrt(variances)

    keep = (counts > 0) & (stds > min_std)
    indicator_mask = nan_mask.any(axis=0)

    info = {
        "means": means,
        "stds": stds,
        "keep": keep,
        "indicator_mask": indicator_mask,
        "add_indicators": add_indicators,
        "n_base": int(keep.sum()),
        "n_indicator": int(indicator_mask.sum()) if add_indicators else 0,
        "dropped": int((~keep).sum()),
        "missing_fraction": float(nan_mask.mean()),
    }

    return _apply_design_matrix(X, info, add_indicators), info


def transform_design_matrix(X, info):
    """Apply previously fitted imputation/scaling to new rows."""
    return _apply_design_matrix(X, info, info["add_indicators"])


def stratified_split(y, val_fraction=0.20, seed=42):
    """Return stratified train/validation index arrays for signed labels.

    Both classes are shuffled with the same RNG and split with the same
    fraction, which preserves the class ratio in both subsets.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for label in (-1, 1):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_fraction)))
        val_idx.append(indices[:n_val])
        train_idx.append(indices[n_val:])
    train_idx = np.sort(np.concatenate(train_idx))
    val_idx = np.sort(np.concatenate(val_idx))
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Metrics (NumPy only).
# ---------------------------------------------------------------------------


def precision_recall_curve(y_true, scores):
    """Precision, recall and thresholds for every distinct score.

    Scores are processed in decreasing order; tied scores are grouped so
    precision/recall are reported once per distinct threshold.

    Returns:
        precision, recall, thresholds: arrays of equal length.
    """
    y_true = np.asarray(y_true).astype(float).ravel()
    scores = np.asarray(scores, dtype=float).ravel()

    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    s_sorted = scores[order]

    cumulative_tp = np.cumsum(y_sorted)
    cumulative_fp = np.cumsum(1.0 - y_sorted)
    total_pos = cumulative_tp[-1]

    # Keep the last index of every group of tied scores.
    group_ends = np.r_[np.flatnonzero(np.diff(s_sorted) != 0), len(s_sorted) - 1]
    tp = cumulative_tp[group_ends]
    fp = cumulative_fp[group_ends]

    thresholds = s_sorted[group_ends]
    precision = tp / (tp + fp)
    recall = tp / total_pos
    return precision, recall, thresholds


def roc_curve(y_true, scores):
    """False-positive and true-positive rates for every distinct score."""
    y_true = np.asarray(y_true).astype(float).ravel()
    scores = np.asarray(scores, dtype=float).ravel()

    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    s_sorted = scores[order]

    cumulative_tp = np.cumsum(y_sorted)
    cumulative_fp = np.cumsum(1.0 - y_sorted)
    total_pos = cumulative_tp[-1]
    total_neg = cumulative_fp[-1]

    group_ends = np.r_[np.flatnonzero(np.diff(s_sorted) != 0), len(s_sorted) - 1]
    tpr = np.r_[0.0, cumulative_tp[group_ends] / total_pos]
    fpr = np.r_[0.0, cumulative_fp[group_ends] / total_neg]
    return fpr, tpr


def _average_ranks(sorted_values):
    """Return 1-based average ranks, averaging over groups of tied values."""
    n = len(sorted_values)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def roc_auc_score(y_true, scores):
    """Area under the ROC curve (Mann-Whitney U statistic, tie-aware)."""
    y_true = np.asarray(y_true).astype(float).ravel()
    scores = np.asarray(scores, dtype=float).ravel()

    n_pos = float(y_true.sum())
    n_neg = float(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = _average_ranks(scores[order])
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision_score(y_true, scores):
    """Average precision using the step (rectangular) interpolation."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    recall_gains = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_gains * precision))


def best_f1_threshold(y_true, scores):
    """Return the threshold that maximises the positive-class F1 and that F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2.0 * precision * recall / (precision + recall)

    if len(thresholds) == 0:
        return 0.5, 0.0
    best = int(np.nanargmax(f1))
    return float(thresholds[best]), float(f1[best])


def classification_metrics(y_true, scores, threshold=0.5):
    """Compute thresholded and ranking metrics for binary probabilities.

    Args:
        y_true: Binary labels in {0, 1}, shape (N,).
        scores: Predicted positive-class probabilities, shape (N,).
        threshold: Decision threshold for the positive class.

    Returns:
        dict with precision, recall, f1, accuracy, roc_auc, average_precision
        and the confusion-matrix counts.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    predictions = (scores >= threshold).astype(int)

    tp = int(np.sum((predictions == 1) & (y_true == 1)))
    tn = int(np.sum((predictions == 0) & (y_true == 0)))
    fp = int(np.sum((predictions == 1) & (y_true == 0)))
    fn = int(np.sum((predictions == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ---------------------------------------------------------------------------
# Reporting helpers (standard library + NumPy only).
# ---------------------------------------------------------------------------


def _format_cell(value):
    """Format a table cell: fixed precision for floats, plain otherwise."""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def format_table(headers, rows):
    """Return an aligned plain-text table for printing in the notebook."""
    headers = [str(header) for header in headers]
    body = [[_format_cell(value) for value in row] for row in rows]
    widths = [
        max([len(headers[col])] + [len(row[col]) for row in body])
        for col in range(len(headers))
    ]
    header_line = "  ".join(
        headers[col].ljust(widths[col]) for col in range(len(headers))
    )
    separator = "  ".join("-" * width for width in widths)
    body_lines = [
        "  ".join(row[col].ljust(widths[col]) for col in range(len(headers)))
        for row in body
    ]
    return "\n".join([header_line, separator, *body_lines])


def write_csv(path, headers, rows):
    """Write a small table to CSV using the standard-library csv module."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([_format_cell(value) for value in row])
    return path


# ---------------------------------------------------------------------------
# Plotting helpers (all figures are saved as vector PDFs).
# ---------------------------------------------------------------------------


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def plot_class_distribution(y, path, title="Class distribution"):
    """Bar chart of the signed label counts."""
    y = np.asarray(y)
    counts = [int((y == -1).sum()), int((y == 1).sum())]
    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(["No CHD (-1)", "CHD (+1)"], counts, color=["#4C72B0", "#C44E52"])
    ax.set_ylabel("Number of samples")
    ax.set_title(title)
    total = sum(counts)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count,
            f"{count}\n({count / total:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.margins(y=0.15)
    return _save(fig, path)


def plot_missingness(X, feature_names, path, top=30):
    """Bar chart of the most missing columns."""
    X = np.asarray(X)
    missing = np.isnan(X).mean(axis=0)
    order = np.argsort(missing)[::-1][:top]
    labels = np.asarray(feature_names)[order]
    values = missing[order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(np.arange(len(values)), values, color="#55A868")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Missing fraction")
    ax.set_title(f"Top {len(values)} columns by missing fraction")
    return _save(fig, path)


def plot_learning_curve(history, path, title="Training history"):
    """Plot the recorded train (and validation) loss against iterations."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    iterations = history["iterations"]
    ax.plot(iterations, history["train_loss"], label="Train loss", color="#4C72B0")
    if history.get("val_loss"):
        ax.plot(
            iterations, history["val_loss"], label="Validation loss", color="#C44E52"
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Weighted cross-entropy")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def plot_roc_curve(y_true, scores, path):
    """Receiver-operating-characteristic curve with the ROC-AUC in the legend."""
    fpr, tpr = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot(fpr, tpr, color="#4C72B0", label=f"NN (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (validation)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_precision_recall_curve(y_true, scores, path):
    """Precision-recall curve with the average precision in the legend."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    baseline = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot(recall, precision, color="#C44E52", label=f"NN (AP = {ap:.3f})")
    ax.axhline(baseline, ls="--", color="grey", label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve (validation)")
    ax.legend(loc="upper right")
    return _save(fig, path)


def plot_threshold_curve(y_true, scores, path, chosen_threshold=None):
    """Precision, recall and F1 as a function of the decision threshold."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2.0 * precision * recall / (precision + recall)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(thresholds, precision, label="Precision", color="#4C72B0")
    ax.plot(thresholds, recall, label="Recall", color="#55A868")
    ax.plot(thresholds, f1, label="F1", color="#C44E52", lw=2)
    if chosen_threshold is not None:
        ax.axvline(chosen_threshold, ls="--", color="black", label="Chosen threshold")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs decision threshold (validation)")
    ax.legend()
    return _save(fig, path)


def plot_confusion_matrix(metrics, path, title="Confusion matrix (validation)"):
    """Heatmap of the confusion matrix stored in ``metrics``."""
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred -1", "Pred +1"])
    ax.set_yticks([0, 1], ["True -1", "True +1"])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            colour = "white" if value > matrix.max() / 2 else "black"
            ax.text(j, i, f"{value}", ha="center", va="center", color=colour)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, path)


def plot_probability_distribution(y_true, scores, path):
    """Histogram of predicted probabilities per true class."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bins = np.linspace(0, 1, 40)
    ax.hist(scores[y_true == 0], bins=bins, alpha=0.6, label="True -1", color="#4C72B0")
    ax.hist(scores[y_true == 1], bins=bins, alpha=0.6, label="True +1", color="#C44E52")
    ax.set_xlabel("Predicted probability of CHD")
    ax.set_ylabel("Number of samples")
    ax.set_yscale("log")
    ax.set_title("Predicted-probability distribution (validation)")
    ax.legend()
    return _save(fig, path)


def save_submission(ids, probabilities, path, threshold=0.5):
    """Write ``Id,Prediction`` predictions in {-1, +1} submission format."""
    ids = np.asarray(ids)
    predictions = np.where(np.asarray(probabilities) >= threshold, 1, -1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Id", "Prediction"])
        for identifier, prediction in zip(ids, predictions):
            writer.writerow([int(identifier), int(prediction)])
    return path
