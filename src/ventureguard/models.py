"""Train-only preprocessing and an sklearn-compatible PyTorch residual MLP.

The neural estimator exports its trained parameters to NumPy for CPU serving;
Torch is required for fitting, not for running the exported estimator.
"""
from __future__ import annotations

import copy
import numpy as np
from scipy.special import expit, ndtr
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score


class SignedLog(TransformerMixin, BaseEstimator):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        return np.sign(x) * np.log1p(np.abs(x))


class MissingMask(TransformerMixin, BaseEstimator):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return np.isnan(np.asarray(x, dtype=float)).astype(float)


class TorchResidualClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, width=64, dropout=.15, lr=.001, weight_decay=.01,
                 balanced=False, max_epochs=100, patience=12, random_state=20261108):
        self.width = width
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.balanced = balanced
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state

    def fit(self, x, y):
        import torch
        from torch import nn
        torch.set_num_threads(2)
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = x.shape[1]
        # Group identical transformed rows for the early-stopping split as well.
        _, groups = np.unique(x, axis=0, return_inverse=True)
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.18, random_state=self.random_state).split(x, y, groups))

        class Block(nn.Module):
            def __init__(self, width, dropout):
                super().__init__()
                self.norm = nn.LayerNorm(width)
                self.a = nn.Linear(width, width)
                self.b = nn.Linear(width, width)
                self.drop = nn.Dropout(dropout)

            def forward(self, z):
                return z + self.b(self.drop(torch.nn.functional.gelu(self.a(self.norm(z)))))

        class Net(nn.Module):
            def __init__(self, n, width, dropout):
                super().__init__()
                self.input = nn.Linear(n, width)
                self.blocks = nn.ModuleList([Block(width, dropout) for _ in range(2)])
                self.output = nn.Linear(width, 1)

            def forward(self, z):
                z = torch.nn.functional.gelu(self.input(z))
                for block in self.blocks:
                    z = block(z)
                return self.output(z).squeeze(1)

        def run(indices, epochs, validation=None):
            torch.manual_seed(self.random_state)
            net = Net(x.shape[1], self.width, self.dropout)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            ratio = float((y[indices] == 0).sum() / max(1, (y[indices] == 1).sum()))
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio if self.balanced else 1.))
            best_ap, best_epoch, best, stale = -1., 1, None, 0
            tx, ty = torch.from_numpy(x), torch.from_numpy(y)
            for epoch in range(epochs):
                net.train()
                for batch in np.array_split(rng.permutation(indices), max(1, int(np.ceil(len(indices) / 256)))):
                    opt.zero_grad()
                    loss = loss_fn(net(tx[batch]), ty[batch])
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5.)
                    opt.step()
                if validation is not None:
                    net.eval()
                    with torch.no_grad():
                        p = torch.sigmoid(net(tx[validation])).numpy()
                    ap = average_precision_score(y[validation], p)
                    if ap > best_ap + 1e-5:
                        best_ap, best_epoch, best, stale = ap, epoch + 1, copy.deepcopy(net.state_dict()), 0
                    else:
                        stale += 1
                    if stale >= self.patience:
                        break
            if validation is not None:
                net.load_state_dict(best)
            return net, best_epoch

        _, self.epochs_ = run(tr, self.max_epochs, va)
        # Refit on all rows given to this estimator using the selected epoch count.
        net, _ = run(np.arange(len(x)), self.epochs_)
        net.eval()
        self.weights_ = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
        with torch.no_grad():
            expected = torch.sigmoid(net(torch.from_numpy(x[:64]))).numpy()
        self.export_max_error_ = float(np.max(np.abs(self.predict_proba(x[:64])[:, 1] - expected)))
        if self.export_max_error_ > 2e-5:
            raise ValueError("NumPy export does not match PyTorch inference")
        return self

    def decision_function(self, x):
        w = self.weights_
        z = np.asarray(x, dtype=np.float64) @ w["input.weight"].T + w["input.bias"]
        z = z * ndtr(z)
        for i in range(2):
            key = f"blocks.{i}."
            v = (z - z.mean(axis=1, keepdims=True)) / np.sqrt(z.var(axis=1, keepdims=True) + 1e-5)
            v = v * w[key + "norm.weight"] + w[key + "norm.bias"]
            v = v @ w[key + "a.weight"].T + w[key + "a.bias"]
            v = v * ndtr(v)
            z = z + v @ w[key + "b.weight"].T + w[key + "b.bias"]
        return (z @ w["output.weight"].T + w["output.bias"]).ravel()

    def predict_proba(self, x):
        p = expit(self.decision_function(x))
        return np.column_stack([1 - p, p])

    def predict(self, x):
        return self.predict_proba(x)[:, 1] >= .5


def candidate_configs(family):
    if family == "logistic":
        return [{"C": c, "balanced": b} for c in [.1, 1.] for b in [False, True]]
    if family == "hgb":
        return [{"leaves": n, "balanced": b} for n in [7, 15] for b in [False, True]]
    return [{"dropout": d, "balanced": b} for d in [.1, .3] for b in [False, True]]


def make_model(family, config, seed, indicators=True, mask_only=False, max_epochs=100, patience=12):
    steps = [("mask", MissingMask())] if mask_only else []
    if family in ["logistic", "mlp"] and not mask_only:
        steps.append(("log", SignedLog()))
    steps.append(("impute", SimpleImputer(strategy="median", add_indicator=indicators and not mask_only, keep_empty_features=True)))
    if family in ["logistic", "mlp"]:
        steps.append(("scale", StandardScaler()))
    weight = "balanced" if config.get("balanced") else None
    if family == "logistic":
        model = LogisticRegression(C=config["C"], class_weight=weight, max_iter=3000)
    elif family == "hgb":
        model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=config["leaves"],
                    min_samples_leaf=25, l2_regularization=5., class_weight=weight,
                    early_stopping=False, random_state=seed)
    else:
        model = TorchResidualClassifier(dropout=config["dropout"], balanced=config["balanced"],
                    max_epochs=max_epochs, patience=patience, random_state=seed)
    return Pipeline(steps + [("model", model)])
