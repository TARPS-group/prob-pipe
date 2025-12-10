from typing import Optional, Callable
import numpy as np
from numpy.typing import NDArray
from probpipe.core.node import Node, wf, WorkflowFunction
from probpipe.core.module import Module


import numpy as np

# Fake distribution classes (for conversion testing)
class Distribution:
    """Base distribution class."""
    @classmethod
    def from_distribution(cls, other, num_samples=1024, **kwargs):
        print(f"[CONVERT] Converting {type(other).__name__} -> {cls.__name__}")
        return cls()

class EmpiricalDistribution(Distribution):
    """Simple empirical distribution."""
    pass



# ==== Regression node implementations ====

class Likelihood(Node):
    @wf
    def log_likelihood(
        self,
        X: np.ndarray,             # for type check
        y: np.ndarray,             # for type check
        sigma: float,           # for type check
        beta: Optional[np.ndarray] = None
    ):
        if beta is None:
            raise ValueError("beta must be provided at call time")
        residual = y - X @ beta
        return -0.5 * np.sum((residual / sigma) ** 2)

class Prior(Node):
    @wf
    def log_prior(
        self,
        prior_mean: np.ndarray, # for type check
        prior_cov: np.ndarray,  # for type check
        beta: Optional[np.ndarray] = None
    ):
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


# Test Inputs

X_train = np.array([[1, 2], [1, 3], [1, 4]])
y_train = np.array([4, 5, 6])
sigma = 1.0
prior_mean = np.zeros(2)
prior_cov = np.eye(2)

# TEST 1 — Dependency type checking (WORKING)

# Injected dependencies (must cover all no-default params)
dependencies = {
    "X": X_train,
    "y": y_train,
    "sigma": sigma,
    "prior_mean": prior_mean,
    "prior_cov": prior_cov,
}

# Instantiate the model (runs dependency type validation)

try:
    print("TEST 1 =>")
    model = RegressionModule(**dependencies)
    print("TEST 1: passed (as expected)")

    print("\nDEPENDENCY GRAPH:")
    model.print_dependency_graph()

except Exception as e:
    print("TEST 1: failed")


# TEST 1.1 — Dependency type checking (type error: expected float)

sigma_bad = "WRONG"  

dependencies_bad = {
    "X": X_train,
    "y": y_train,
    "sigma": sigma_bad,
    "prior_mean": prior_mean,
    "prior_cov": prior_cov,
}

try:
    print("\nTEST 1.1 =>")
    model_bad = RegressionModule(**dependencies_bad)
    print("TEST 1.1: passed")
except Exception as e:
    print("TEST 1.1: failed (as expected)")
    print(f"ERROR: {e}")


# TEST 2 - Workflow-time distribution conversion

beta_emp = EmpiricalDistribution()

# If we add conversion in WorkflowFunction to accept distr -> NDArray, ...
# For now this will error unless we implemented conversion logic.

try:
    print("\nTEST 2 =>")
    print("Running fit() with distribution-type beta:")
    out = model.fit(beta=beta_emp)
    print("Log posterior:", out)
    print("TEST 2: passed")

except Exception as e:
    print("TEST 2: failed (as expected)")
    print("ERROR during fit:", e)


# TEST 3 — Normal computation with correct ndarray beta

beta = np.array([1.0, 1.0])
X_test = np.array([[1, 5]])

try:
    print("\nTEST 3 =>")

    model = RegressionModule(**dependencies)

    print("Running fit() with correct ndarray beta:")
    log_post = model.fit(beta=beta)
    print("Log posterior:", log_post)

    print("\nRunning predict():")
    pred = model.predict(beta=beta, X_new=X_test)
    print("Prediction:", pred)

    print("TEST 3: passed")


except Exception as e:
    print("TEST 3: failed")
    print(f"ERROR: {e}")


# TEST 4 — Normal computation with correct ndarray beta (with Prefect)


try:
    print("\nTEST 4 =>")

    model = RegressionModule(backend="prefect", **dependencies)

    print("Running fit() with correct ndarray beta:")
    log_post = model.fit(beta=beta)
    print("\nLog posterior:", log_post)

    print("\nRunning predict():")
    pred = model.predict(beta=beta, X_new=X_test)
    print("\nPrediction:", pred)

    print("TEST 4: passed")


except Exception as e:
    print("TEST 4: failed")
    print(f"ERROR: {e}")
