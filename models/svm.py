# models/svm.py
import numpy as np

def train_linear_svm_sgd(X, y, C=1.0, lr=0.01, epochs=100, batch_size=32, verbose=False):
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0
    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(n_samples)
        X_shuf = X[idx]; y_shuf = y[idx]
        for start in range(0, n_samples, batch_size):
            xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]
            margins = yb * (xb.dot(w) + b)
            mask = margins < 1.0
            if np.any(mask):
                grad_w_hinge = -C * np.sum((yb[mask][:, None] * xb[mask]), axis=0)
                grad_b_hinge = -C * np.sum(yb[mask])
            else:
                grad_w_hinge = np.zeros_like(w); grad_b_hinge = 0.0
            grad_w = w + grad_w_hinge
            grad_b = grad_b_hinge
            w -= lr * grad_w
            b -= lr * grad_b
        if verbose and (epoch % max(1, epochs//10) == 0):
            margins_all = y * (X.dot(w) + b)
            hinge_losses = np.maximum(0, 1 - margins_all)
            obj = 0.5 * np.dot(w, w) + C * np.sum(hinge_losses)
            print(f"[SVM] epoch {epoch}/{epochs} obj={obj:.4f} avg_hinge={np.mean(hinge_losses):.4f}")
    return w, b

def train_ovr_svm(X, y, C=0.5, lr=0.01, epochs=200, batch_size=64, verbose=False):
    classes = np.unique(y)
    models = {}
    for cls in classes:
        y_bin = np.where(y==cls, 1, -1)
        if verbose:
            print(f"Training SVM for class {cls}: +1 count={np.sum(y_bin==1)}, -1 count={np.sum(y_bin==-1)}")
        w, b = train_linear_svm_sgd(X, y_bin, C=C, lr=lr, epochs=epochs, batch_size=batch_size, verbose=verbose)
        models[cls] = (w,b)
    return models, list(classes)

def predict_ovr_svm(models, classes, X):
    scores = np.vstack([X.dot(models[c][0]) + models[c][1] for c in classes]).T
    idx = np.argmax(scores, axis=1)
    return np.array([classes[i] for i in idx])
