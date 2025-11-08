# models/random_forest.py
import numpy as np
from collections import Counter
import pandas as pd
from models.decision_tree import DecisionTreeClassifierCustom

class RandomForestCustom:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_total_features = X.shape[1]
        if self.n_features is None:
            self.n_features = int(np.sqrt(n_total_features)) if n_total_features>0 else 1
        for _ in range(self.n_trees):
            idxs = np.random.choice(len(X), len(X), replace=True)
            X_samp, y_samp = X[idxs], y[idxs]
            feat_idxs = np.random.choice(n_total_features, self.n_features, replace=False)
            tree = DecisionTreeClassifierCustom(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_samp[:, feat_idxs], y_samp)
            self.trees.append((tree, feat_idxs))

    def predict(self, X):
        if len(self.trees) == 0:
            return np.zeros(X.shape[0], dtype=int)
        all_preds = []
        for tree, feat_idxs in self.trees:
            preds = tree.predict(X[:, feat_idxs])
            all_preds.append(preds)
        all_preds = np.array(all_preds).T
        y_pred = []
        for sample_preds in all_preds:
            y_pred.append(Counter(sample_preds).most_common(1)[0][0])
        return np.array(y_pred, dtype=int)

def permutation_feature_importance(rf_model, X_test, y_test, feature_names, n_repeats=1):
    baseline_acc = np.mean(rf_model.predict(X_test) == y_test)
    importances = []
    for i in range(X_test.shape[1]):
        accs = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            np.random.shuffle(X_perm[:, i])
            accs.append(np.mean(rf_model.predict(X_perm) == y_test))
        mean_perm_acc = np.mean(accs)
        importances.append(baseline_acc - mean_perm_acc)
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False).reset_index(drop=True)
    return fi_df
