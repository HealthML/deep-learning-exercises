# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "torch>=2.3",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    return mo, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 4: Implement, check, and investigate a two-layer MLP

    This notebook gives you the experiment scaffolding. You implement the small pieces that matter: the NumPy forward pass, the NumPy backward pass, finite-difference gradient checking, and one parameter update.

    ## Workflow

    For each implementation step:

    1. read the equations;
    2. write the smallest implementation you can;
    3. run the checkpoint below it; and
    4. use the evidence from the checks before changing anything else.

    The plotting, data generation, interactive controls, and training orchestration are provided so that your time goes into the model and the reasoning.

    ## Learning goals

    By the end, you should be able to:

    - implement a two-layer MLP with a nonlinear hidden layer;
    - derive and implement its gradients;
    - use finite differences and PyTorch autograd to check those gradients;
    - distinguish capacity problems from optimization problems; and
    - transfer the implementation to a new dataset.
    """)
    return


@app.cell
def _(np):
    X_XOR = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    Y_XOR = np.array([[0.0, 1.0, 1.0, 0.0]])
    return X_XOR, Y_XOR


@app.cell(hide_code=True)
def _(np):
    def initialize(width, input_dim=2, seed=9):
        rng = np.random.default_rng(seed)
        return [
            rng.normal(0.0, 0.5, (width, input_dim)),
            np.zeros((width, 1)),
            rng.normal(0.0, 0.5, (1, width)),
            np.zeros((1, 1)),
        ]

    def train_linear(x, y, steps=3000, lr=0.1):
        w = np.zeros((1, x.shape[0]))
        b = np.zeros((1, 1))
        history = []
        n = x.shape[1]
        for _ in range(steps):
            pred = w @ x + b
            err = pred - y
            history.append(np.sum(err**2) / (2 * n))
            delta = err / n
            w -= lr * delta @ x.T
            b -= lr * np.sum(delta, axis=1, keepdims=True)
        return [w, b], np.asarray(history)

    def linear_forward(params, x):
        return params[0] @ x + params[1]

    def accuracy(pred, y):
        return np.mean((pred >= 0.5) == (y >= 0.5))

    def plot_boundary(
        ax, forward_fn, params, x, y, activation="tanh", pad=0.45
    ):
        x_min, x_max = x[0].min() - pad, x[0].max() + pad
        y_min, y_max = x[1].min() - pad, x[1].max() + pad
        gx, gy = np.meshgrid(
            np.linspace(x_min, x_max, 120), np.linspace(y_min, y_max, 120)
        )
        grid = np.vstack([gx.ravel(), gy.ravel()])
        scores = forward_fn(params, grid, activation)[0].reshape(gx.shape)
        ax.contourf(
            gx,
            gy,
            scores,
            levels=np.linspace(0.0, 1.0, 11),
            cmap="RdBu",
            alpha=0.3,
        )
        if scores.min() < 0.5 < scores.max():
            ax.contour(
                gx, gy, scores, levels=[0.5], colors="black", linewidths=1.5
            )
        ax.scatter(
            x[0], x[1], c=y.ravel(), cmap="RdBu", edgecolors="black", s=70
        )
        ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max), aspect="equal")

    def make_moons(n=160, noise=0.1, seed=7):
        rng = np.random.default_rng(seed)
        half = n // 2
        theta = rng.uniform(0.0, np.pi, half)
        first = np.vstack([np.cos(theta), np.sin(theta)])
        second = np.vstack([1.0 - np.cos(theta), 0.5 - np.sin(theta)])
        x = np.hstack([first, second])
        x += rng.normal(0.0, noise, x.shape)
        y = np.hstack([np.zeros((1, half)), np.ones((1, half))])
        return x, y

    return (
        accuracy,
        initialize,
        linear_forward,
        make_moons,
        plot_boundary,
        train_linear,
    )


@app.cell(hide_code=True)
def _(X_XOR, Y_XOR, accuracy, linear_forward, mo, plt, train_linear):
    mo.md(
        r"""
    ## 1. Baseline: why does a linear model fail on XOR?

    This part is provided as a baseline. Predict the result before running it:

    > Can an affine map classify all four XOR points at threshold 0.5?
            """
    )
    _linear_params, _linear_history = train_linear(X_XOR, Y_XOR)
    _linear_predictions = linear_forward(_linear_params, X_XOR)
    _linear_accuracy = accuracy(_linear_predictions, Y_XOR)

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.5))
    _axes[0].plot(_linear_history)
    _axes[0].set(
        xlabel="parameter updates",
        ylabel="mean half-squared error",
        title="Training loss",
    )
    _axes[1].scatter(
        X_XOR[0],
        X_XOR[1],
        c=_linear_predictions.ravel(),
        cmap="RdBu",
        vmin=0,
        vmax=1,
        edgecolors="black",
        s=100,
    )
    for _i, _prediction in enumerate(_linear_predictions.ravel()):
        _axes[1].annotate(
            f"{_prediction:.2f}", (X_XOR[0, _i] + 0.04, X_XOR[1, _i] + 0.04)
        )
    _axes[1].set(
        xlabel="$x_1$",
        ylabel="$x_2$",
        title="Linear predictions",
        xticks=[0, 1],
        yticks=[0, 1],
    )
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(
                f"**Observed:** final loss `{_linear_history[-1]:.3f}`, threshold accuracy `{_linear_accuracy:.0%}`. Predictions: `{_linear_predictions.ravel().round(3).tolist()}`"
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Implement the forward pass

    Use the column-vector convention:

    $$
    H = W_1X+b_1,\qquad A=\phi(H),\qquad \hat Y=W_2A+b_2.
    $$

    Complete `forward_student` below. It should return:

    ```python
    prediction, (preactivation, hidden_activation)
    ```

    Start with $\phi$ = `tanh`. Add ReLU only after the tanh checkpoint works.
    """)
    return


@app.function
def forward_student(params, x, activation="tanh"):
    _w1, _b1, _w2, _b2 = params
    # TODO: compute the hidden preactivation H.
    # TODO: apply tanh (and optionally ReLU) to obtain A.
    # TODO: compute the linear output Y_hat.
    return None


@app.cell(hide_code=True)
def _(X_XOR, Y_XOR, initialize, mo, np):
    _checkpoint_params = initialize(width=2, seed=9)
    _forward_result = forward_student(_checkpoint_params, X_XOR)
    if _forward_result is None:
        _forward_checkpoint_view = mo.callout(
            "Complete `forward_student` before continuing.", kind="warn"
        )
        _forward_ready = False
    else:
        _checkpoint_prediction, (_checkpoint_h, _checkpoint_a) = (
            _forward_result
        )
        _forward_ready = (
            _checkpoint_prediction.shape == Y_XOR.shape
            and _checkpoint_h.shape == (2, X_XOR.shape[1])
            and _checkpoint_a.shape == (2, X_XOR.shape[1])
            and np.isfinite(_checkpoint_prediction).all()
        )
        _message = (
            "Forward-pass checkpoint passed."
            if _forward_ready
            else "Check the forward-pass shapes and values."
        )
        _forward_checkpoint_view = mo.callout(
            _message, kind="success" if _forward_ready else "warn"
        )
    _forward_checkpoint_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Implement the loss and backward pass

    Use the mean half-squared error:

    $$
    L=\frac{1}{2n}\sum_i(\hat y_i-y_i)^2.
    $$

    For the tanh network, implement:

    $$
    \delta_2=\frac{\hat Y-Y}{n},\qquad
    \delta_1=(W_2^T\delta_2)\odot(1-A^2),
    $$

    $$
    \nabla_{W_2}=\delta_2A^T,\quad
    \nabla_{b_2}=\sum_i\delta_{2,i},\quad
    \nabla_{W_1}=\delta_1X^T,\quad
    \nabla_{b_1}=\sum_i\delta_{1,i}.
    $$

    Return `(loss, [grad_W1, grad_b1, grad_W2, grad_b2])`. Keep the gradient order identical to the parameter order.
    """)
    return


@app.function
def loss_and_grads_student(params, x, y, activation="tanh"):
    # TODO: call forward_student and unpack its cached values.
    # TODO: compute the loss and output delta.
    # TODO: apply the activation derivative to get delta1.
    # TODO: return the four gradients in parameter order.
    return None


@app.cell(hide_code=True)
def _(X_XOR, Y_XOR, initialize, mo, np):
    _backward_params = initialize(width=2, seed=9)
    _backward_result = loss_and_grads_student(_backward_params, X_XOR, Y_XOR)
    if _backward_result is None:
        _backward_checkpoint_view = mo.callout(
            "Complete `loss_and_grads_student` before continuing.", kind="warn"
        )
        _backward_ready = False
    else:
        _loss_value, _backward_grads = _backward_result
        _expected_shapes = [(2, 2), (2, 1), (1, 2), (1, 1)]
        _backward_ready = (
            np.isscalar(_loss_value)
            and len(_backward_grads) == 4
            and [gradient.shape for gradient in _backward_grads]
            == _expected_shapes
            and all(
                np.isfinite(gradient).all() for gradient in _backward_grads
            )
        )
        _message = (
            "Backward-pass checkpoint passed."
            if _backward_ready
            else "Check the loss, gradient order, and gradient shapes."
        )
        _backward_checkpoint_view = mo.callout(
            _message, kind="success" if _backward_ready else "warn"
        )
    _backward_checkpoint_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Implement finite-difference gradient checking

    For one parameter, estimate the derivative with:

    $$
    \frac{\partial L}{\partial\theta_i}\approx
    \frac{L(\theta_i+\varepsilon)-L(\theta_i-\varepsilon)}{2\varepsilon}.
    $$

    Implement `gradient_check_student` by comparing this estimate with the analytic gradients from `loss_and_grads_student`. Loop over every parameter array and every index. Return the maximum normalized discrepancy.
    """)
    return


@app.function
def gradient_check_student(params, x, y, activation="tanh", eps=1e-5):
    # TODO: compute analytic gradients once.
    # TODO: perturb one scalar parameter at a time, in both directions.
    # TODO: restore the original value after each estimate.
    # TODO: return the largest normalized discrepancy.
    return None


@app.cell(hide_code=True)
def _(X_XOR, Y_XOR, initialize, mo):
    _finite_difference_result = gradient_check_student(
        initialize(2, seed=4), X_XOR, Y_XOR
    )
    if _finite_difference_result is None:
        _finite_difference_view = mo.callout(
            "Complete `gradient_check_student` before continuing.", kind="warn"
        )
    else:
        _finite_difference_view = mo.callout(
            f"Finite-difference check returned maximum normalized discrepancy `{_finite_difference_result:.3e}`.",
            kind="success" if _finite_difference_result < 1e-6 else "warn",
        )
    _finite_difference_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Implement one parameter update

    Complete the update below. It should return a new parameter list or the updated list:

    $$
    \theta\leftarrow\theta-\eta\nabla_\theta L.
    $$

    Do not change the gradient function while testing this step.
    """)
    return


@app.function
def apply_update_student(params, grads, learning_rate):
    # TODO: update every parameter using the old parameter and its gradient.
    # Keep the order [W1, b1, W2, b2].
    return None


@app.cell(hide_code=True)
def _(initialize, mo, np):
    _update_params = initialize(width=2, seed=9)
    _update_grads = [np.ones_like(_parameter) for _parameter in _update_params]
    _updated = apply_update_student(_update_params, _update_grads, 0.1)
    _update_ready = (
        _updated is not None
        and len(_updated) == 4
        and all(
            np.asarray(_new).shape == _old.shape
            for _old, _new in zip(_update_params, _updated)
        )
    )
    mo.callout(
        "Update checkpoint passed."
        if _update_ready
        else "Complete `apply_update_student` before training.",
        kind="success" if _update_ready else "warn",
    )
    return


@app.cell(hide_code=True)
def _(X_XOR, Y_XOR, initialize, mo, np):
    def train_student(
        x, y, width=2, seed=9, steps=2000, learning_rate=0.1, activation="tanh"
    ):
        _params = initialize(width, x.shape[0], seed)
        _history = []
        for _ in range(steps):
            _result = loss_and_grads_student(_params, x, y, activation)
            if _result is None:
                return None, None
            _loss, _grads = _result
            _history.append(_loss)
            _params = apply_update_student(_params, _grads, learning_rate)
            if _params is None:
                return None, None
        return _params, np.asarray(_history)

    _all_ready = all(
        value
        for value in (
            forward_student(initialize(2), X_XOR) is not None,
            loss_and_grads_student(initialize(2), X_XOR, Y_XOR) is not None,
            gradient_check_student(initialize(2), X_XOR, Y_XOR) is not None,
            apply_update_student(
                initialize(2), [np.zeros_like(_p) for _p in initialize(2)], 0.1
            )
            is not None,
        )
    )
    mo.md(
        "**Implementation status:** all required student functions are available; continue to the experiments."
        if _all_ready
        else "**Implementation status:** complete the TODO cells above before running the experiments."
    )
    return (train_student,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Use PyTorch autograd as an independent reference

    Finite differences are useful, but PyTorch gives us another comparison: the same forward equations evaluated by an automatic differentiation engine.

    The cell below is provided. Study the correspondence between the NumPy equations and `loss.backward()`.
    """)
    return


@app.cell
def _(initialize, torch):
    def torch_autodiff_grads(params, x, y, activation="tanh"):
        _tensors = [
            torch.tensor(parameter, dtype=torch.float64, requires_grad=True)
            for parameter in params
        ]
        _w1, _b1, _w2, _b2 = _tensors
        _torch_x = torch.tensor(x, dtype=torch.float64)
        _torch_y = torch.tensor(y, dtype=torch.float64)
        _h = _w1 @ _torch_x + _b1
        _a = torch.tanh(_h) if activation == "tanh" else torch.relu(_h)
        _prediction = _w2 @ _a + _b2
        _loss = ((_prediction - _torch_y) ** 2).mean() / 2
        _loss.backward()
        return float(_loss.detach()), [
            parameter.grad.detach().numpy().copy() for parameter in _tensors
        ]

    def torch_gradcheck(x, y, activation="tanh"):
        _torch_x = torch.tensor(x, dtype=torch.float64)
        _torch_y = torch.tensor(y, dtype=torch.float64)

        def _loss_function(_w1, _b1, _w2, _b2):
            _h = _w1 @ _torch_x + _b1
            _a = torch.tanh(_h) if activation == "tanh" else torch.relu(_h)
            _prediction = _w2 @ _a + _b2
            return ((_prediction - _torch_y) ** 2).mean() / 2

        _parameters = tuple(
            torch.tensor(parameter, dtype=torch.float64, requires_grad=True)
            for parameter in initialize(2, x.shape[0], seed=4)
        )
        return torch.autograd.gradcheck(
            _loss_function,
            _parameters,
            eps=1e-6,
            atol=1e-5,
            rtol=1e-3,
        )

    return torch_autodiff_grads, torch_gradcheck


@app.cell(hide_code=True)
def _(X_XOR, Y_XOR, initialize, mo, np, torch_autodiff_grads, torch_gradcheck):
    mo.md(
        r"""
    ### Compare the two gradient implementations

    The manual NumPy gradients and PyTorch gradients should agree closely. This is the important comparison for your implementation.

    `torch.autograd.gradcheck` is a separate demonstration: it checks PyTorch's own graph with finite differences. It does **not** directly check your NumPy function.
            """
    )
    _torch_params = initialize(2, seed=4)
    _torch_loss, _torch_grads = torch_autodiff_grads(
        _torch_params, X_XOR, Y_XOR
    )
    _torch_gradcheck_result = torch_gradcheck(X_XOR, Y_XOR)
    _manual_result = loss_and_grads_student(_torch_params, X_XOR, Y_XOR)
    if _manual_result is None:
        _torch_comparison_view = mo.callout(
            f"PyTorch reference ready. `torch.autograd.gradcheck`: `{_torch_gradcheck_result}`. Complete the NumPy backward pass to compare gradients.",
            kind="warn",
        )
        _torch_comparison_error = None
    else:
        _manual_loss, _manual_grads = _manual_result
        _torch_comparison_error = max(
            np.max(np.abs(_manual_gradient - _torch_gradient))
            for _manual_gradient, _torch_gradient in zip(
                _manual_grads, _torch_grads
            )
        )
        _torch_comparison_view = mo.md(
            f"PyTorch `gradcheck`: `{_torch_gradcheck_result}`. Maximum absolute difference between NumPy and PyTorch gradients: `{_torch_comparison_error:.3e}`."
        )
    _torch_comparison_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 7. Investigate: capacity versus optimization

    Now use your implementation. Change one setting at a time in the controls below. Compare several seeds before making a claim about width or activation.
            """
    )
    experiment_width = mo.ui.slider(
        start=1, stop=8, step=1, value=2, label="hidden width"
    )
    experiment_activation = mo.ui.dropdown(
        options=["tanh", "relu"], value="tanh", label="activation"
    )
    experiment_learning_rate = mo.ui.slider(
        start=0.01, stop=0.5, step=0.01, value=0.1, label="learning rate"
    )
    experiment_steps = mo.ui.slider(
        start=200, stop=4000, step=200, value=2000, label="updates"
    )
    experiment_seed = mo.ui.slider(
        start=0, stop=20, step=1, value=9, label="initialization seed"
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    experiment_width,
                    experiment_activation,
                    experiment_learning_rate,
                ]
            ),
            mo.hstack([experiment_steps, experiment_seed]),
        ]
    )
    return (
        experiment_activation,
        experiment_learning_rate,
        experiment_seed,
        experiment_steps,
        experiment_width,
    )


@app.cell(hide_code=True)
def _(
    X_XOR,
    Y_XOR,
    accuracy,
    experiment_activation,
    experiment_learning_rate,
    experiment_seed,
    experiment_steps,
    experiment_width,
    mo,
    plot_boundary,
    plt,
    train_student,
):
    _experiment_params, _experiment_history = train_student(
        X_XOR,
        Y_XOR,
        width=experiment_width.value,
        seed=experiment_seed.value,
        steps=experiment_steps.value,
        learning_rate=experiment_learning_rate.value,
        activation=experiment_activation.value,
    )
    if _experiment_params is None:
        experiment_view = mo.callout(
            "Complete the student implementation cells before running this experiment.",
            kind="warn",
        )
    else:
        _experiment_predictions = forward_student(
            _experiment_params, X_XOR, experiment_activation.value
        )[0]
        _experiment_accuracy = accuracy(_experiment_predictions, Y_XOR)
        _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.5))
        _axes[0].plot(_experiment_history)
        _axes[0].set(
            xlabel="parameter updates",
            ylabel="mean half-squared error",
            title="Training loss",
        )
        plot_boundary(
            _axes[1],
            forward_student,
            _experiment_params,
            X_XOR,
            Y_XOR,
            experiment_activation.value,
            pad=0.35,
        )
        _axes[1].set(
            title="Learned decision boundary",
            xlabel="$x_1$",
            ylabel="$x_2$",
            xticks=[0, 1],
            yticks=[0, 1],
        )
        _fig.tight_layout()
        experiment_view = mo.vstack(
            [
                mo.md(
                    f"**Result:** width `{experiment_width.value}`, `{experiment_activation.value}`, seed `{experiment_seed.value}` → final loss `{_experiment_history[-1]:.4f}`, accuracy `{_experiment_accuracy:.0%}`, predictions `{_experiment_predictions.ravel().round(3).tolist()}`"
                ),
                _fig,
            ]
        )
    experiment_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A real debugging task

    Make a copy of your backward-pass code and deliberately introduce one of these bugs:

    - remove the hidden activation derivative;
    - transpose the outer product for `grad_W1`; or
    - use the wrong mean factor.

    Then use your gradient checker and the loss curve to identify the bug. Do not rely on array shapes alone.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 8. Transfer: apply the same implementation to two moons

    XOR has only four observations. Use the same functions on a noisy two-moons dataset. Predict first: will a width-eight tanh network still learn a nonlinear boundary?
            """
    )
    moon_noise = mo.ui.slider(
        start=0.0, stop=0.35, step=0.05, value=0.1, label="moon noise"
    )
    moon_seed = mo.ui.slider(
        start=0, stop=20, step=1, value=7, label="dataset seed"
    )
    mo.hstack([moon_noise, moon_seed])
    return moon_noise, moon_seed


@app.cell(hide_code=True)
def _(
    accuracy,
    make_moons,
    mo,
    moon_noise,
    moon_seed,
    plot_boundary,
    plt,
    train_student,
):
    _moon_x, _moon_y = make_moons(noise=moon_noise.value, seed=moon_seed.value)
    _moon_params, _moon_history = train_student(
        _moon_x,
        _moon_y,
        width=8,
        seed=9,
        steps=2500,
        learning_rate=0.1,
        activation="tanh",
    )
    if _moon_params is None:
        moon_view = mo.callout(
            "Complete the student implementation cells before running this experiment.",
            kind="warn",
        )
    else:
        _moon_predictions = forward_student(_moon_params, _moon_x, "tanh")[0]
        _moon_accuracy = accuracy(_moon_predictions, _moon_y)
        _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.5))
        plot_boundary(
            _axes[0],
            forward_student,
            _moon_params,
            _moon_x,
            _moon_y,
            "tanh",
            pad=0.35,
        )
        _axes[0].set(
            title="Two-moons decision boundary", xlabel="$x_1$", ylabel="$x_2$"
        )
        _axes[1].plot(_moon_history)
        _axes[1].set(
            xlabel="parameter updates",
            ylabel="mean half-squared error",
            title="Training loss",
        )
        _fig.tight_layout()
        moon_view = mo.vstack(
            [
                mo.md(
                    f"**Transfer result:** noise `{moon_noise.value:.2f}`, seed `{moon_seed.value}` → accuracy `{_moon_accuracy:.0%}`, final loss `{_moon_history[-1]:.4f}`."
                ),
                _fig,
            ]
        )
    moon_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final reflection

    Use evidence from the notebook:

    1. Why did the affine model fail on XOR while the MLP could succeed?
    2. Which observations support a capacity explanation, and which support an optimization explanation?
    3. How did finite differences and PyTorch autograd complement each other?
    4. What did your deliberate bug look like in the gradient check and in the loss curve?
    5. What would you change before using this MLP for a real classification problem?
    """)
    return


if __name__ == "__main__":
    app.run()
