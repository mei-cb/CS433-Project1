"""Implementation of the methods required for EPFL Machine Learning Project 1."""

import numpy as np


def compute_loss(y, tx, w):
    """Compute the mean squared error loss with a factor 1/2."""
    e = y - tx @ w
    return 0.5 * np.mean(e**2)


def compute_gradient(y, tx, w):
    """Compute the gradient of the mean squared error loss."""
    e = y - tx @ w
    return -(tx.T @ e) / len(y)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: Labels, shape (N,).
        tx: Feature matrix, shape (N, D).
        initial_w: Initial weight vector, shape (D,).
        max_iters: Number of gradient-descent iterations.
        gamma: Step size.

    Returns:
        w: Final weight vector.
        loss: MSE loss at the final weights.
    """
    w = initial_w.copy()

    for _ in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w -= gamma * gradient

    loss = compute_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent.

    Each iteration uses one randomly selected training example.

    Args:
        y: Labels, shape (N,).
        tx: Feature matrix, shape (N, D).
        initial_w: Initial weight vector, shape (D,).
        max_iters: Number of SGD iterations.
        gamma: Step size.

    Returns:
        w: Final weight vector.
        loss: MSE loss at the final weights.
    """
    w = initial_w.copy()
    n_samples = len(y)

    for _ in range(max_iters):
        i = np.random.randint(n_samples)

        x_i = tx[i]
        y_i = y[i]

        error = y_i - x_i @ w
        gradient = -x_i * error

        w -= gamma * gradient

    loss = compute_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Compute the least-squares solution.

    Uses the normal equations:
        w = (X^T X)^(-1) X^T y

    Args:
        y: Labels, shape (N,).
        tx: Feature matrix, shape (N, D).

    Returns:
        w: Least-squares weight vector.
        loss: MSE loss at the solution.
    """
    a = tx.T @ tx
    b = tx.T @ y

    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """Compute the ridge-regression solution.

    Args:
        y: Labels, shape (N,).
        tx: Feature matrix, shape (N, D).
        lambda_: Ridge regularization parameter.

    Returns:
        w: Ridge-regression weight vector.
        loss: Unregularized MSE loss at the solution.
    """
    n_samples = tx.shape[0]

    a = tx.T @ tx + 2 * n_samples * lambda_ * np.eye(tx.shape[1])
    b = tx.T @ y

    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)

    return w, loss


def sigmoid(t):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-t))


def logistic_loss(y, tx, w):
    """Compute the average logistic loss for labels in {0, 1}."""
    logits = tx @ w

    # log(1 + exp(logits)) - y * logits
    #
    return np.mean(np.logaddexp(0, logits) - y * logits)


def logistic_gradient(y, tx, w):
    """Compute the gradient of the logistic loss."""
    probabilities = sigmoid(tx @ w)
    return tx.T @ (probabilities - y) / len(y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.

    Args:
        y: Binary labels in {0, 1}, shape (N,).
        tx: Feature matrix, shape (N, D).
        initial_w: Initial weight vector, shape (D,).
        max_iters: Number of gradient-descent iterations.
        gamma: Step size.

    Returns:
        w: Final weight vector.
        loss: Logistic loss at the final weights.
    """
    w = initial_w.copy()

    for _ in range(max_iters):
        gradient = logistic_gradient(y, tx, w)
        w -= gamma * gradient

    loss = logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    Args:
        y: Binary labels in {0, 1}, shape (N,).
        tx: Feature matrix, shape (N, D).
        lambda_: Regularization parameter.
        initial_w: Initial weight vector, shape (D,).
        max_iters: Number of gradient-descent iterations.
        gamma: Step size.

    Returns:
        w: Final weight vector.
        loss: Unregularized logistic loss at the final weights.
    """
    w = initial_w.copy()

    for _ in range(max_iters):
        gradient = logistic_gradient(y, tx, w)

        # Regularization gradient.
        gradient += 2 * lambda_ * w

        w -= gamma * gradient

    loss = logistic_loss(y, tx, w)
    return w, loss


def relu(x):
    """ReLU activation function."""
    return np.maximum(0.0, x)


def relu_gradient(x):
    """Derivative of ReLU."""
    return (x > 0).astype(float)


def sigmoid_stable(t):
    """Numerically stable sigmoid."""
    result = np.empty_like(t, dtype=float)

    positive = t >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-t[positive]))

    exp_t = np.exp(t[~positive])
    result[~positive] = exp_t / (1.0 + exp_t)

    return result


def neural_network_loss(y, probabilities):
    """Binary cross-entropy loss."""
    eps = 1e-15
    probabilities = np.clip(probabilities, eps, 1.0 - eps)

    return -np.mean(y * np.log(probabilities) + (1.0 - y) * np.log(1.0 - probabilities))


def _neural_network_init(dims, rng):
    """He-initialise a fully connected network with layer sizes ``dims``.

    ``dims`` is a list ``[n_features, h1, ..., hk, 1]``. The first ``k``
    layers use ReLU activations and the output layer uses a sigmoid, so the
    network predicts a probability for the positive class.

    Returns:
        params: dict mapping ``"W{i}"``/``"b{i}"`` to the weights and biases
            of layer ``i`` (1-based).
    """
    params = {}
    for layer, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:]), start=1):
        std = np.sqrt(2.0 / fan_in)
        params[f"W{layer}"] = rng.normal(0.0, std, size=(fan_in, fan_out))
        params[f"b{layer}"] = np.zeros(fan_out)
    return params


def _neural_network_forward(tx, params):
    """Forward pass through the network.

    Args:
        tx: Input features, shape (N, D).
        params: Network parameters as produced by :func:`_neural_network_init`.

    Returns:
        activations: List of layer activations, starting with ``tx``.
        pre_activations: List of pre-activation values for each layer.
    """
    n_layers = len(params) // 2
    activations = [tx]
    pre_activations = []

    for layer in range(1, n_layers + 1):
        z = activations[-1] @ params[f"W{layer}"] + params[f"b{layer}"]
        pre_activations.append(z)
        if layer == n_layers:
            activations.append(sigmoid_stable(z))
        else:
            activations.append(relu(z))

    return activations, pre_activations


def _neural_network_backward(y, activations, pre_activations, params, pos_weight):
    """Backpropagation for weighted binary cross-entropy.

    Args:
        y: Binary labels in {0, 1}, shape (N,).
        activations: Forward-pass activations.
        pre_activations: Forward-pass pre-activations.
        params: Network parameters.
        pos_weight: Weight applied to positive examples (class imbalance).

    Returns:
        grads: dict mapping ``"dW{i}"``/``"db{i}"`` to parameter gradients.
    """
    n_layers = len(params) // 2
    n_batch = len(y)
    y = y.reshape(-1, 1)
    weights = np.where(y == 1, pos_weight, 1.0)

    # Derivative of the weighted cross-entropy w.r.t. the output logits.
    delta = weights * (activations[-1] - y) / n_batch

    grads = {}
    for layer in range(n_layers, 0, -1):
        grads[f"dW{layer}"] = activations[layer - 1].T @ delta
        grads[f"db{layer}"] = np.sum(delta, axis=0)

        if layer > 1:
            delta = (delta @ params[f"W{layer}"].T) * relu_gradient(
                pre_activations[layer - 2]
            )

    return grads


def _weighted_bce(y, probabilities, pos_weight):
    """Weighted binary cross-entropy used as the training objective."""
    y = np.asarray(y).reshape(-1, 1)
    eps = 1e-15
    probabilities = np.clip(probabilities.reshape(-1, 1), eps, 1.0 - eps)
    weights = np.where(y == 1, pos_weight, 1.0)
    losses = -(y * np.log(probabilities) + (1.0 - y) * np.log(1.0 - probabilities))
    return float(np.mean(weights * losses))


def neural_network_predict(tx, params):
    """Forward pass returning predicted probabilities.

    Supports an arbitrary number of dense ReLU layers followed by a sigmoid
    output layer, so it works with both the shallow network returned by
    :func:`neural_network` and the deeper networks returned by
    :func:`neural_network_train`.

    Args:
        tx: Feature matrix, shape (N, D).
        params: dict mapping ``"W{i}"``/``"b{i}"`` to layer parameters.

    Returns:
        probabilities: Predicted positive-class probabilities, shape (N,).
    """
    activations, _ = _neural_network_forward(tx, params)
    return activations[-1].ravel()


def neural_network_train(
    y,
    tx,
    hidden_dims=(32,),
    max_iters=1000,
    gamma=0.001,
    batch_size=2048,
    seed=42,
    l2=0.0,
    pos_weight=1.0,
    optimizer="adam",
    record_every=50,
    valid=None,
    verbose=False,
):
    """Train a multi-layer perceptron with mini-batch stochastic optimisation.

    The network is a stack of dense ReLU layers followed by a sigmoid output
    and is trained by minimising the weighted binary cross-entropy

        L = -1/N * sum_i a_i [ y_i log p_i + (1 - y_i) log(1 - p_i) ]
            + l2 * ||W||_2^2,

    where ``a_i = pos_weight`` for positive examples and ``a_i = 1`` otherwise.
    This weighting is the main lever against the strong class imbalance of the
    dataset.

    Args:
        y: Binary labels in {0, 1}, shape (N,).
        tx: Feature matrix, shape (N, D). Should be standardised beforehand.
        hidden_dims: Sequence with the size of each hidden layer.
        max_iters: Number of optimisation iterations (mini-batches).
        gamma: Learning rate.
        batch_size: Mini-batch size.
        seed: Seed for the reproducible mini-batch shuffling.
        l2: L2 weight-decay coefficient (biases are not penalised).
        pos_weight: Weight of positive examples in the loss.
        optimizer: ``"adam"`` or ``"sgd"``.
        record_every: Record the train/validation loss every this many iters.
        valid: Optional ``(y_val, tx_val)`` tuple for validation monitoring.
        verbose: Print the loss at every recorded iteration.

    Returns:
        params: Trained network parameters.
        history: dict with keys ``"iterations"``, ``"train_loss"`` and
            ``"val_loss"`` (the latter only when ``valid`` is provided).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float).reshape(-1)

    if optimizer not in {"adam", "sgd"}:
        raise ValueError("optimizer must be 'adam' or 'sgd'.")

    dims = [tx.shape[1], *hidden_dims, 1]
    params = _neural_network_init(dims, rng)
    n_layers = len(dims) - 1

    # Adam moment estimates.
    first_moment = {key: np.zeros_like(value) for key, value in params.items()}
    second_moment = {key: np.zeros_like(value) for key, value in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    history = {"iterations": [], "train_loss": [], "val_loss": []}
    n_samples = tx.shape[0]

    for iteration in range(1, max_iters + 1):
        batch_indices = rng.choice(
            n_samples, size=min(batch_size, n_samples), replace=False
        )
        X_batch = tx[batch_indices]
        y_batch = y[batch_indices]

        activations, pre_activations = _neural_network_forward(X_batch, params)
        grads = _neural_network_backward(
            y_batch, activations, pre_activations, params, pos_weight
        )

        for layer in range(1, n_layers + 1):
            key_w, key_b = f"W{layer}", f"b{layer}"
            grad_w = grads[f"dW{layer}"] + 2.0 * l2 * params[key_w]
            grad_b = grads[f"db{layer}"]

            if optimizer == "adam":
                first_moment[key_w] = beta1 * first_moment[key_w] + (1 - beta1) * grad_w
                second_moment[key_w] = (
                    beta2 * second_moment[key_w] + (1 - beta2) * grad_w**2
                )
                first_moment[key_b] = beta1 * first_moment[key_b] + (1 - beta1) * grad_b
                second_moment[key_b] = (
                    beta2 * second_moment[key_b] + (1 - beta2) * grad_b**2
                )

                m_hat_w = first_moment[key_w] / (1 - beta1**iteration)
                v_hat_w = second_moment[key_w] / (1 - beta2**iteration)
                m_hat_b = first_moment[key_b] / (1 - beta1**iteration)
                v_hat_b = second_moment[key_b] / (1 - beta2**iteration)

                params[key_w] -= gamma * m_hat_w / (np.sqrt(v_hat_w) + eps)
                params[key_b] -= gamma * m_hat_b / (np.sqrt(v_hat_b) + eps)
            else:
                params[key_w] -= gamma * grad_w
                params[key_b] -= gamma * grad_b

        if iteration % record_every == 0 or iteration == max_iters:
            train_probabilities = neural_network_predict(tx, params)
            train_loss = _weighted_bce(y, train_probabilities, pos_weight)
            history["iterations"].append(iteration)
            history["train_loss"].append(train_loss)

            if valid is not None:
                y_val, tx_val = valid
                val_probabilities = neural_network_predict(tx_val, params)
                val_loss = _weighted_bce(
                    np.asarray(y_val), val_probabilities, pos_weight
                )
                history["val_loss"].append(val_loss)

            if verbose:
                message = f"iter {iteration:6d} | train loss {train_loss:.5f}"
                if valid is not None:
                    message += f" | val loss {history['val_loss'][-1]:.5f}"
                print(message)

    if valid is None:
        history.pop("val_loss")

    return params, history


def neural_network(
    y,
    tx,
    hidden_dim=32,
    max_iters=1000,
    gamma=0.001,
    batch_size=2048,
    seed=42,
):
    """Train a one-hidden-layer neural network using mini-batch SGD.

    This is the shallow, SGD-only variant kept for backward compatibility with
    the earlier exploratory notebook. For the improved trainer (Adam, deeper
    architectures, L2 regularisation, class weighting and validation
    monitoring) use :func:`neural_network_train`.

    Args:
        y: Binary labels in {0, 1}, shape (N,).
        tx: Feature matrix, shape (N, D).
        hidden_dim: Number of units in the hidden ReLU layer.
        max_iters: Number of SGD iterations (mini-batches).
        gamma: Step size.
        batch_size: Mini-batch size.
        seed: Seed for the reproducible mini-batch shuffling.

    Returns:
        params: Final network parameters.
        losses: Training loss recorded every 50 iterations.
    """
    params, history = neural_network_train(
        y,
        tx,
        hidden_dims=(hidden_dim,),
        max_iters=max_iters,
        gamma=gamma,
        batch_size=batch_size,
        seed=seed,
        optimizer="sgd",
        record_every=50,
    )
    return params, history["train_loss"]
