from typing import Optional, Callable
import numpy as np
from numpy.typing import NDArray
from probpipe.core.node import Node, wf, WorkflowFunction
from probpipe.core.module import Module


import numpy as np


# ==== Regression node implementations ====
import numpy as np

class Likelihood(Node):
    @wf
    def log_likelihood(self, X, y, sigma, beta=None):
        if beta is None:
            raise ValueError("beta must be provided at call time")
        residual = y - X @ beta
        return -0.5 * np.sum((residual / sigma) ** 2)

class Prior(Node):
    @wf
    def log_prior(self, prior_mean, prior_cov, beta=None):
        if beta is None:
            raise ValueError("beta must be provided at call time")
        diff = beta - prior_mean
        cov_inv = np.linalg.inv(prior_cov)
        return -0.5 * diff @ cov_inv @ diff

class RegressionModule(Module):
    @classmethod
    def _define_nodes(cls, **dependencies):
        return {
            "likelihood": Likelihood(**dependencies),
            "prior": Prior(**dependencies),
        }

    @wf
    def fit(self, beta):
        ll = self._nodes["likelihood"].log_likelihood(beta=beta)
        lp = self._nodes["prior"].log_prior(beta=beta)
        return ll + lp

    @wf
    def predict(self, beta, X_new):
        return X_new @ beta


# === Usage ===

X_train = np.array([[1, 2], [1, 3], [1, 4]])
y_train = np.array([4, 5, 6])
sigma = 1.0
prior_mean = np.zeros(2)
prior_cov = np.eye(2)

# Injected dependencies (must cover all no-default params)
dependencies = {
    "X": X_train,
    "y": y_train,
    "sigma": sigma,
    "prior_mean": prior_mean,
    "prior_cov": prior_cov,
}

model = RegressionModule(**dependencies)

beta = np.array([1.0, 1.0])

log_post = model.fit(beta=beta)
print("Log posterior:", log_post)

X_test = np.array([[1, 5]])
pred = model.predict(beta=beta, X_new=X_test)
print("Prediction:", pred)