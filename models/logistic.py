# models/logistic.py
import numpy as np

class LogisticRegressionCustom:
    """
    Multiclass Logistic Regression (softmax) from scratch with L2 regularization
    and mini-batch gradient descent.

    API:
      - fit(X, y): X shape (n_samples, n_features), y int labels [0..K-1]
      - predict(X): returns np.array of predicted class indices
    """

    def __init__(self, lr=0.02, reg_lambda=0.001, epochs=500, batch_size=32, decay=0.0, random_state=42):
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        self.decay = decay
        self.random_state = random_state
        self.W = None  # (n_features+1, n_classes) including bias as last feature

    def _add_bias(self, X):
        n = X.shape[0]
        ones = np.ones((n, 1), dtype=X.dtype)
        return np.hstack([X, ones])

    def _one_hot(self, y, n_classes):
        oh = np.zeros((y.shape[0], n_classes), dtype=np.float64)
        oh[np.arange(y.shape[0]), y] = 1.0
        return oh

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = classes.size

        # Ensure labels are 0..K-1
        label_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([label_to_idx[int(t)] for t in y], dtype=int)

        Xb = self._add_bias(X)
        # Initialize weights small
        self.W = rng.normal(loc=0.0, scale=0.01, size=(n_features + 1, n_classes))

        for epoch in range(self.epochs):
            # Shuffle
            perm = rng.permutation(n_samples)
            Xb_shuf = Xb[perm]
            y_shuf = y_idx[perm]

            # Learning rate decay (time-based)
            lr_t = self.lr / (1.0 + self.decay * epoch)

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                xb = Xb_shuf[start:end]
                yb = y_shuf[start:end]

                # Forward: logits and softmax
                logits = xb @ self.W  # (B, C)
                logits -= logits.max(axis=1, keepdims=True)  # stability
                exp = np.exp(logits)
                probs = exp / (np.sum(exp, axis=1, keepdims=True) + 1e-12)

                Y = self._one_hot(yb, n_classes)  # (B, C)

                # Gradients (mean over batch) + L2 on non-bias weights
                grad = (xb.T @ (probs - Y)) / xb.shape[0]  # (D+1, C)
                # Do not regularize bias row (last row)
                reg = self.reg_lambda * self.W
                reg[-1, :] = 0.0
                grad += reg

                # Update
                self.W -= lr_t * grad

        # Store classes mapping
        self.classes_ = classes
        return self

    def predict(self, X):
        if self.W is None:
            raise RuntimeError("Model is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        Xb = self._add_bias(X)
        logits = Xb @ self.W
        idx = np.argmax(logits, axis=1)
        return self.classes_[idx].astype(int)
