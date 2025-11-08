# models/decision_tree.py
import numpy as np
from collections import Counter

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf(self):
        return self.value is not None

class DecisionTreeClassifierCustom:
    """
    Decision tree from scratch.
    Parameters:
      max_depth, min_samples_split, min_samples_leaf, criterion ('gini' or 'entropy'), random_state
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion='gini', random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.root = None
        if random_state is not None:
            np.random.seed(random_state)

    def gini(self, y):
        if len(y) == 0: return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum([p**2 for p in proportions])

    def entropy(self, y):
        if len(y) == 0: return 0
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def information_gain(self, parent, left_child, right_child):
        if self.criterion == 'entropy':
            parent_impurity = self.entropy(parent)
        else:
            parent_impurity = self.gini(parent)
        n = len(parent)
        n_left = len(left_child)
        n_right = len(right_child)
        if n_left == 0 or n_right == 0:
            return 0
        if self.criterion == 'entropy':
            child_impurity = (n_left / n) * self.entropy(left_child) + (n_right / n) * self.entropy(right_child)
        else:
            child_impurity = (n_left / n) * self.gini(left_child) + (n_right / n) * self.gini(right_child)
        return parent_impurity - child_impurity

    def split(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]
        for feature in range(n_features):
            unique_vals = np.unique(X[:, feature])
            thresholds = unique_vals
            for threshold in thresholds:
                left_X, left_y, right_X, right_y = self.split(X, y, feature, threshold)
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                gain = self.information_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold, best_gain

    def build_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))
        if (self.max_depth is not None and depth >= self.max_depth) or n_classes == 1 or n_samples < self.min_samples_split:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionNode(value=leaf_value)
        best_feature, best_threshold, best_gain = self.best_split(X, y)
        if best_feature is None or best_gain == 0:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionNode(value=leaf_value)
        left_X, left_y, right_X, right_y = self.split(X, y, best_feature, best_threshold)
        left_subtree = self.build_tree(left_X, left_y, depth + 1)
        right_subtree = self.build_tree(right_X, right_y, depth + 1)
        return DecisionNode(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)
        return self

    def predict_sample(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

    def predict(self, X):
        return np.array([self.predict_sample(x, self.root) for x in X])

    def get_depth(self, node=None):
        if node is None: node = self.root
        if node.is_leaf(): return 0
        return 1 + max(self.get_depth(node.left), self.get_depth(node.right))

    def get_n_leaves(self, node=None):
        if node is None: node = self.root
        if node.is_leaf(): return 1
        return self.get_n_leaves(node.left) + self.get_n_leaves(node.right)
