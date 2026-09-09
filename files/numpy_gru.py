"""
Minimal GRU classifier, hand-rolled in NumPy with manual backprop-through-
time (BPTT). Chosen over a full LSTM as a deliberately simpler recurrent
cell (3 gates vs 4, no separate cell state) — same role in the SENTINEL
pipeline (sequence -> phase probabilities), much less gradient bookkeeping.

Architecture: GRU(hidden_dim) -> Dropout -> Linear(n_classes) -> Softmax
Trained with mini-batch Adam + cross-entropy loss.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class NumpyGRUClassifier:
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        def init(n_in, n_out):
            return rng.normal(0, np.sqrt(1.0 / n_in), size=(n_in, n_out))

        h = hidden_dim
        # gate weights: z (update), r (reset), n/hh (candidate)
        self.params = {
            "Wxz": init(input_dim, h), "Whz": init(h, h), "bz": np.zeros(h),
            "Wxr": init(input_dim, h), "Whr": init(h, h), "br": np.zeros(h),
            "Wxh": init(input_dim, h), "Whh": init(h, h), "bh": np.zeros(h),
            "Wy": init(h, n_classes), "by": np.zeros(n_classes),
        }
        self._adam_m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._adam_v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

        self.mean_ = None
        self.std_ = None

    # ---------------- forward ----------------
    def _forward(self, X: np.ndarray, dropout_p: float, rng: np.random.Generator | None):
        """
        X: (batch, seq_len, input_dim), already normalized.
        Returns logits (batch, n_classes), probs, and cache for backward.
        """
        p = self.params
        batch, seq_len, _ = X.shape
        h = np.zeros((batch, self.hidden_dim))
        cache = []
        for t in range(seq_len):
            x_t = X[:, t, :]
            z = _sigmoid(x_t @ p["Wxz"] + h @ p["Whz"] + p["bz"])
            r = _sigmoid(x_t @ p["Wxr"] + h @ p["Whr"] + p["br"])
            hh = np.tanh(x_t @ p["Wxh"] + (r * h) @ p["Whh"] + p["bh"])
            h_new = (1 - z) * h + z * hh
            cache.append((x_t, h, z, r, hh))
            h = h_new

        if dropout_p > 0 and rng is not None:
            mask = (rng.random(h.shape) > dropout_p).astype(h.dtype) / (1 - dropout_p)
        else:
            mask = np.ones_like(h)
        h_drop = h * mask

        logits = h_drop @ p["Wy"] + p["by"]
        probs = _softmax(logits)
        return logits, probs, cache, h, mask

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xn = (X - self.mean_) / self.std_
        _, probs, _, _, _ = self._forward(Xn, dropout_p=0.0, rng=None)
        return probs

    # ---------------- backward ----------------
    def _backward(self, X, y_onehot, probs, cache, h_final, drop_mask):
        """Standard (unweighted) backward pass — kept for gradient-check tests."""
        dlogits = (probs - y_onehot) / X.shape[0]
        return self._backward_from_dlogits(X, dlogits, cache, h_final, drop_mask)

    def _backward_from_dlogits(self, X, dlogits, cache, h_final, drop_mask):
        """Backward pass starting from an already-computed dlogits (allows
        per-sample loss weighting to be applied by the caller before BPTT)."""
        p = self.params
        grads = {k: np.zeros_like(v) for k, v in p.items()}

        grads["Wy"] = (h_final * drop_mask).T @ dlogits
        grads["by"] = dlogits.sum(axis=0)
        dh_drop = dlogits @ p["Wy"].T
        dh_next = dh_drop * drop_mask

        for t in reversed(range(len(cache))):
            x_t, h_prev, z, r, hh = cache[t]

            dh_new = dh_next
            dz = dh_new * (hh - h_prev)
            dhh = dh_new * z
            dh_prev = dh_new * (1 - z)

            dtanh = dhh * (1 - hh ** 2)
            grads["bh"] += dtanh.sum(axis=0)
            grads["Wxh"] += x_t.T @ dtanh
            rh = r * h_prev
            grads["Whh"] += rh.T @ dtanh
            d_rh = dtanh @ p["Whh"].T
            dr = d_rh * h_prev
            dh_prev += d_rh * r

            dsig_z = dz * z * (1 - z)
            grads["bz"] += dsig_z.sum(axis=0)
            grads["Wxz"] += x_t.T @ dsig_z
            grads["Whz"] += h_prev.T @ dsig_z
            dh_prev += dsig_z @ p["Whz"].T

            dsig_r = dr * r * (1 - r)
            grads["br"] += dsig_r.sum(axis=0)
            grads["Wxr"] += x_t.T @ dsig_r
            grads["Whr"] += h_prev.T @ dsig_r
            dh_prev += dsig_r @ p["Whr"].T

            dh_next = dh_prev  # propagate to t-1

        return grads

    def _adam_step(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8, clip=5.0):
        self._t += 1
        for k in self.params:
            g = np.clip(grads[k], -clip, clip)
            self._adam_m[k] = beta1 * self._adam_m[k] + (1 - beta1) * g
            self._adam_v[k] = beta2 * self._adam_v[k] + (1 - beta2) * (g ** 2)
            m_hat = self._adam_m[k] / (1 - beta1 ** self._t)
            v_hat = self._adam_v[k] / (1 - beta2 ** self._t)
            self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 3e-3,
        dropout_p: float = 0.3,
        val_fraction: float = 0.2,
        class_weight: bool = True,
        seed: int = 0,
        verbose: bool = False,
    ):
        """
        groups: optional array (len == len(X)) identifying which scenario
            each window came from. If given, the train/val split is done
            BY GROUP (whole scenarios held out), not by individual window —
            required because overlapping sliding windows from the same
            scenario are near-duplicates and leak badly across a random
            per-window split.
        class_weight: if True, weight the cross-entropy loss by inverse
            class frequency (computed on the training split) to counter
            the natural NORMAL-heavy class imbalance in escalation data.
        """
        rng = np.random.default_rng(seed)
        self.mean_ = X.reshape(-1, X.shape[-1]).mean(axis=0)
        self.std_ = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-8
        Xn = (X - self.mean_) / self.std_

        n = len(Xn)
        if groups is not None:
            uniq_groups = rng.permutation(np.unique(groups))
            n_val_groups = max(1, int(len(uniq_groups) * val_fraction))
            val_groups = set(uniq_groups[:n_val_groups].tolist())
            val_mask = np.isin(groups, list(val_groups))
            train_idx = np.where(~val_mask)[0]
            val_idx = np.where(val_mask)[0]
        else:
            idx = rng.permutation(n)
            n_val = max(1, int(n * val_fraction))
            val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_train, y_train = Xn[train_idx], y[train_idx]
        X_val, y_val = Xn[val_idx], y[val_idx]
        y_train_oh = np.eye(self.n_classes)[y_train]

        if class_weight:
            counts = np.bincount(y_train, minlength=self.n_classes).astype(float)
            counts = np.maximum(counts, 1.0)
            w = counts.sum() / (self.n_classes * counts)  # inverse-frequency
            sample_w_train = w[y_train]
        else:
            sample_w_train = np.ones(len(y_train))

        n_train = len(X_train)
        for epoch in range(epochs):
            perm = rng.permutation(n_train)
            X_train, y_train_oh, y_train = X_train[perm], y_train_oh[perm], y_train[perm]
            sample_w_train = sample_w_train[perm]
            losses = []
            for start in range(0, n_train, batch_size):
                xb = X_train[start:start + batch_size]
                yb = y_train_oh[start:start + batch_size]
                wb = sample_w_train[start:start + batch_size]
                if len(xb) == 0:
                    continue
                logits, probs, cache, h_final, mask = self._forward(xb, dropout_p, rng)
                per_sample_loss = -np.sum(yb * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
                loss = np.mean(per_sample_loss * wb)
                losses.append(loss)
                # weighted dlogits: scale each sample's gradient contribution
                # by its class weight before averaging over the batch.
                dlogits = (probs - yb) * wb[:, None] / len(xb)
                grads = self._backward_from_dlogits(xb, dlogits, cache, h_final, mask)
                self._adam_step(grads, lr)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                # NOTE: X_val is already normalized (sliced from Xn above) —
                # use _forward directly, NOT predict_proba, which would
                # normalize a second time and silently corrupt the eval.
                _, val_probs, _, _, _ = self._forward(X_val, dropout_p=0.0, rng=None)
                val_pred = val_probs.argmax(axis=1)
                val_acc = float(np.mean(val_pred == y_val))
                print(f"epoch {epoch:4d}  train_loss={np.mean(losses):.4f}  val_acc={val_acc:.3f}  "
                      f"n_train={n_train} n_val={len(X_val)}")
        return self
