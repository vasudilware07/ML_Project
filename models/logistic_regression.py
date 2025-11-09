# models/logistic_regression.py
"""
Logistic Regression (One-vs-Rest) - from scratch using mini-batch SGD.
Supports multi-class by training one binary logistic model per class (OvR).
"""

import numpy as np

class BinaryLogistic:
    def __init__(self, lr=0.01, epochs=100, batch_size=32, C=1.0, verbose=False, random_state=None):
        """
        Binary logistic regression (sigmoid) with L2 regularization.
        C: regularization multiplier (like SVM's C). We implement reg as (1/2)||w||^2 and multiply loss by C -> consistent with earlier SVM usage.
        """
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.C = C
        self.verbose = verbose
        self.random_state = random_state
        self.w = None
        self.b = 0.0

        if random_state is not None:
            np.random.seed(random_state)

    def _sigmoid(self, z):
        # numerically stable sigmoid
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _loss_and_grads(self, Xb, yb):
        """
        Returns (loss, grad_w, grad_b) for minibatch
        We use binary cross-entropy: -[y log(p) + (1-y) log(1-p)]
        Total objective = 0.5 * ||w||^2 + C * sum BCE
        """
        n = Xb.shape[0]
        logits = Xb.dot(self.w) + self.b
        probs = self._sigmoid(logits)
        # clip probs to avoid log(0)
        eps = 1e-12
        probs = np.clip(probs, eps, 1 - eps)
        # BCE
        bce = - (yb * np.log(probs) + (1 - yb) * np.log(1 - probs))
        loss_data = np.mean(bce)
        # total loss: reg + C * data_loss
        loss = 0.5 * np.dot(self.w, self.w) + self.C * loss_data

        # gradients
        # gradient of BCE part: (p - y) / n
        diff = (probs - yb) / n
        grad_w = self.w + self.C * (Xb.T.dot(diff))
        grad_b = self.C * np.sum(diff)
        return loss, grad_w, grad_b

    def fit(self, X, y):
        """
        X: (n_samples, n_features), float
        y: binary labels {0,1}
        """
        n_samples, n_features = X.shape
        # init
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

        for epoch in range(1, self.epochs + 1):
            idx = np.random.permutation(n_samples)
            X_shuf = X[idx]
            y_shuf = y[idx]
            for start in range(0, n_samples, self.batch_size):
                xb = X_shuf[start:start + self.batch_size]
                yb = y_shuf[start:start + self.batch_size]
                if xb.shape[0] == 0:
                    continue
                _, grad_w, grad_b = self._loss_and_grads(xb, yb)
                # parameter update
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

            if self.verbose and (epoch % max(1, self.epochs // 10) == 0):
                # compute loss over whole data for logging
                logits = X.dot(self.w) + self.b
                probs = self._sigmoid(logits)
                eps = 1e-12
                probs = np.clip(probs, eps, 1-eps)
                bce = - (y * np.log(probs) + (1 - y) * np.log(1 - probs))
                loss_data = np.mean(bce)
                obj = 0.5 * np.dot(self.w, self.w) + self.C * loss_data
                print(f"[Logistic] epoch {epoch}/{self.epochs} obj={obj:.4f} avg_bce={loss_data:.4f}")

        return self

    def predict_proba(self, X):
        logits = X.dot(self.w) + self.b
        return self._sigmoid(logits)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


class LogisticOVR:
    """
    One-vs-Rest logistic regression wrapper for multiclass classification.
    Trains one BinaryLogistic per class (class vs rest).
    """
    def __init__(self, lr=0.01, epochs=100, batch_size=32, C=1.0, verbose=False, random_state=None):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.C = C
        self.verbose = verbose
        self.random_state = random_state
        self.models = {}      # class_label -> BinaryLogistic
        self.classes_ = None

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = list(classes)
        for cls in classes:
            y_bin = (y == cls).astype(int)
            model = BinaryLogistic(lr=self.lr, epochs=self.epochs, batch_size=self.batch_size,
                                   C=self.C, verbose=self.verbose, random_state=self.random_state)
            model.fit(X, y_bin)
            self.models[cls] = model
        return self

    def predict(self, X):
        # compute decision score for each class (use probability as score)
        scores = []
        for cls in self.classes_:
            probs = self.models[cls].predict_proba(X)  # shape (n_samples,)
            scores.append(probs)
        scores = np.vstack(scores).T  # (n_samples, n_classes)
        idx = np.argmax(scores, axis=1)
        return np.array([self.classes_[i] for i in idx])

    def predict_proba(self, X):
        probs = np.vstack([self.models[cls].predict_proba(X) for cls in self.classes_]).T
        return probs
