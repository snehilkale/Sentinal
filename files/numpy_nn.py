"""
Hand-rolled NumPy autoencoder (no torch available in this environment).

Architecture per spec: encoder input_dim -> 64 -> 32 -> 8, mirrored
decoder 8 -> 32 -> 64 -> input_dim. Trained with plain mini-batch SGD
(Adam) and MSE loss, manual forward/backward pass.

Kept intentionally small/fast (see training budget decision): this is
enough to demonstrate the architecture separates NORMAL reconstruction
error from escalating-phase error, not a tuned production model.
"""

from __future__ import annotations

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)


class Dense:
    """A single fully-connected layer with Adam optimizer state."""

    def __init__(self, n_in: int, n_out: int, activation: str, rng: np.random.Generator):
        # He init for ReLU layers, small Xavier-ish for linear output layer.
        scale = np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)
        self.W = rng.normal(0, scale, size=(n_in, n_out)).astype(np.float64)
        self.b = np.zeros(n_out, dtype=np.float64)
        self.activation = activation

        # Adam state
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t = 0

        self._last_input = None
        self._last_z = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_input = x
        z = x @ self.W + self.b
        self._last_z = z
        return _relu(z) if self.activation == "relu" else z

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            dz = dout * _relu_grad(self._last_z)
        else:
            dz = dout
        dW = self._last_input.T @ dz / dz.shape[0]
        db = dz.mean(axis=0)
        dx = dz @ self.W.T
        self._dW, self._db = dW, db
        return dx

    def step(self, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for param, grad, m_attr, v_attr in (
            ("W", self._dW, "mW", "vW"),
            ("b", self._db, "mb", "vb"),
        ):
            m = getattr(self, m_attr)
            v = getattr(self, v_attr)
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** self.t)
            v_hat = v / (1 - beta2 ** self.t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, param, getattr(self, param) - update)


class NumpyAutoencoder:
    """Symmetric autoencoder: input -> 64 -> 32 -> 8 -> 32 -> 64 -> input."""

    def __init__(self, input_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.layers = [
            Dense(input_dim, 64, "relu", rng),
            Dense(64, 32, "relu", rng),
            Dense(32, 8, "relu", rng),      # bottleneck
            Dense(8, 32, "relu", rng),
            Dense(32, 64, "relu", rng),
            Dense(64, input_dim, "linear", rng),
        ]
        self.input_dim = input_dim
        self.mean_ = None
        self.std_ = None
        self.threshold_ = None

    def _forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def _backward(self, dout: np.ndarray):
        d = dout
        for layer in reversed(self.layers):
            d = layer.backward(d)

    def fit(
        self,
        X_normal: np.ndarray,
        epochs: int = 300,
        batch_size: int = 64,
        lr: float = 1e-3,
        val_fraction: float = 0.15,
        seed: int = 0,
        verbose: bool = False,
    ) -> "NumpyAutoencoder":
        rng = np.random.default_rng(seed)
        self.mean_ = X_normal.mean(axis=0)
        self.std_ = X_normal.std(axis=0) + 1e-8
        Xn = (X_normal - self.mean_) / self.std_

        n = len(Xn)
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_fraction))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        X_train, X_val = Xn[train_idx], Xn[val_idx]

        n_train = len(X_train)
        for epoch in range(epochs):
            perm = rng.permutation(n_train)
            X_train = X_train[perm]
            losses = []
            for start in range(0, n_train, batch_size):
                batch = X_train[start:start + batch_size]
                if len(batch) == 0:
                    continue
                recon = self._forward(batch)
                diff = recon - batch
                loss = float(np.mean(diff ** 2))
                losses.append(loss)
                dout = 2 * diff / diff.shape[0]  # d(MSE)/d(recon)
                self._backward(dout)
                for layer in self.layers:
                    layer.step(lr)
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                val_recon = self._forward(X_val)
                val_loss = float(np.mean((val_recon - X_val) ** 2))
                print(f"epoch {epoch:4d}  train_mse={np.mean(losses):.5f}  val_mse={val_loss:.5f}")

        # Set threshold at 95th percentile of per-sample reconstruction
        # error on the held-out validation split (per spec).
        val_recon = self._forward(X_val)
        val_errors = np.mean((val_recon - X_val) ** 2, axis=1)
        self.threshold_ = float(np.percentile(val_errors, 95))
        self._val_error_ref = val_errors  # kept for normalization reference
        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Raw per-sample MSE reconstruction error (in normalized-input space)."""
        Xn = (X - self.mean_) / self.std_
        recon = self._forward(Xn)
        return np.mean((recon - Xn) ** 2, axis=1)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruction error normalized to roughly [0, 1]: error == threshold
        (95th pct of NORMAL validation error) maps to 0.5, scaled so that
        ~2x threshold maps close to 1.0. Values are clipped to [0, 1].
        """
        err = self.reconstruction_error(X)
        scaled = err / (self.threshold_ + 1e-12)
        # smooth squashing centered at ratio=1 (i.e. == threshold)
        out = 1.0 / (1.0 + np.exp(-3.0 * (scaled - 1.0)))
        return np.clip(out, 0.0, 1.0)
