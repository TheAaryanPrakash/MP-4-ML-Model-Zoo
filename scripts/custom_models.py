"""
Small sklearn-compatible implementations of two classic-but-rarely-packaged
ML techniques, used to round out the model zoo beyond what's on PyPI.

Neither of these is deep learning: ELM has a single random (untrained)
hidden layer with a closed-form ridge-regression readout (no backprop, no
iterative gradient descent), and the SOM classifier is a competitive-learning
lookup table, not a differentiable network.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder


class ExtremeLearningMachineClassifier(BaseEstimator, ClassifierMixin):
    """
    Extreme Learning Machine (Huang et al., 2006): a single hidden layer with
    fixed random weights (never trained) feeding a closed-form ridge-regression
    output layer. No backpropagation -- the whole "training" step is one
    linear solve, which is the entire point of the technique.
    """

    def __init__(self, n_hidden=512, alpha=1.0, random_state=0):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.random_state = random_state

    def _hidden(self, X):
        H = X @ self.W_ + self.b_
        return np.tanh(H)

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        self.classes_ = np.unique(y)
        self.W_ = rng.uniform(-1, 1, size=(n_features, self.n_hidden))
        self.b_ = rng.uniform(-1, 1, size=(self.n_hidden,))

        enc = OneHotEncoder(sparse_output=False)
        Y = enc.fit_transform(y.reshape(-1, 1))
        self._encoder = enc

        H = self._hidden(X)
        # ridge-regularized closed-form solution: beta = (H^T H + aI)^-1 H^T Y
        n_h = H.shape[1]
        beta = np.linalg.solve(
            H.T @ H + self.alpha * np.eye(n_h), H.T @ Y
        )
        self.beta_ = beta
        return self

    def predict(self, X):
        H = self._hidden(X)
        scores = H @ self.beta_
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]


class SOMClassifier(BaseEstimator, ClassifierMixin):
    """
    Self-Organizing Map used as a classifier: train an unsupervised SOM on
    X, then label each map unit by the majority class of training samples
    whose best-matching unit (BMU) it is. Predictions look up the BMU of
    each test sample. Falls back to the overall majority class for any BMU
    never visited during training.
    """

    def __init__(self, grid_size=10, sigma=1.5, learning_rate=0.5,
                 n_iter=2000, random_state=0):
        self.grid_size = grid_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        from minisom import MiniSom

        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        som = MiniSom(
            self.grid_size, self.grid_size, n_features,
            sigma=self.sigma, learning_rate=self.learning_rate,
            random_seed=self.random_state,
        )
        som.random_weights_init(X)
        som.train_random(X, self.n_iter)
        self.som_ = som

        unit_votes = {}
        for xi, yi in zip(X, y):
            unit = som.winner(xi)
            unit_votes.setdefault(unit, []).append(yi)

        self.unit_labels_ = {
            unit: np.bincount(votes).argmax()
            for unit, votes in unit_votes.items()
        }
        self.majority_class_ = np.bincount(y).argmax()
        return self

    def predict(self, X):
        preds = np.empty(X.shape[0], dtype=self.classes_.dtype)
        for i, xi in enumerate(X):
            unit = self.som_.winner(xi)
            preds[i] = self.unit_labels_.get(unit, self.majority_class_)
        return preds
