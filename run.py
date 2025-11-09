# run.py
import os
import time
import pickle
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_and_preprocess, StandardScalerCustom, classification_report_custom
from models.knn import KNN
from models.decision_tree import DecisionTreeClassifierCustom
from models.random_forest import RandomForestCustom, permutation_feature_importance
from models.svm import train_ovr_svm, predict_ovr_svm
from models.logistic import LogisticRegressionCustom

sns.set(style="whitegrid")
np.random.seed(42)

OUTPUT_DIR = "outputs"
ARTIFACTS = "artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS, exist_ok=True)

def stratified_train_test_split(X_df, y_arr, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    classes = np.unique(y_arr)
    train_idx = []
    test_idx = []
    for cls in classes:
        cls_idx = np.where(y_arr==cls)[0]
        np.random.shuffle(cls_idx)
        n_test = max(1, int(len(cls_idx)*test_size))
        test_idx.extend(cls_idx[:n_test])
        train_idx.extend(cls_idx[n_test:])
    return X_df.iloc[train_idx], X_df.iloc[test_idx], y_arr[train_idx], y_arr[test_idx]

def save_plot(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path

def run_all_models(csv_path):
    print("Loading & preprocessing...")
    X_enc, y, feature_names, X_raw, label_map = load_and_preprocess(csv_path)
    X_train_df, X_test_df, y_train, y_test = stratified_train_test_split(X_enc, y, test_size=0.2, random_state=42)
    scaler = StandardScalerCustom()
    X_train = scaler.fit_transform(X_train_df.values.astype(float))
    X_test = scaler.transform(X_test_df.values.astype(float))

    results = OrderedDict()

    # KNN
    print("Training KNN...")
    t0 = time.time()
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    t1 = time.time()
    y_pred_knn = knn.predict(X_test)
    acc_knn = np.mean(y_test == y_pred_knn)
    report_knn, _, _, f1_knn, _ = classification_report_custom(y_test, y_pred_knn, class_names=[k for k,v in sorted(label_map.items(), key=lambda x:x[1])])
    results['KNN'] = {'acc':acc_knn, 'time': t1-t0, 'y_pred':y_pred_knn, 'report':report_knn, 'f1':f1_knn}

    # Decision Tree
    print("Training Decision Tree...")
    t0 = time.time()
    dt = DecisionTreeClassifierCustom(max_depth=10, min_samples_split=2, min_samples_leaf=1, criterion='gini', random_state=42)
    dt.fit(X_train, y_train)
    t1 = time.time()
    y_pred_dt = dt.predict(X_test)
    acc_dt = np.mean(y_test == y_pred_dt)
    report_dt, _, _, f1_dt, _ = classification_report_custom(y_test, y_pred_dt, class_names=[k for k,v in sorted(label_map.items(), key=lambda x:x[1])])
    results['DecisionTree'] = {'acc':acc_dt, 'time': t1-t0, 'y_pred':y_pred_dt, 'report':report_dt, 'f1':f1_dt}

    # Random Forest
    print("Training Random Forest...")
    t0 = time.time()
    rf = RandomForestCustom(n_trees=100, max_depth=8, min_samples_split=5, n_features=None)
    rf.fit(X_train, y_train)
    t1 = time.time()
    y_pred_rf = rf.predict(X_test)
    acc_rf = np.mean(y_test == y_pred_rf)
    report_rf, _, _, f1_rf, _ = classification_report_custom(y_test, y_pred_rf, class_names=[k for k,v in sorted(label_map.items(), key=lambda x:x[1])])
    results['RandomForest'] = {'acc':acc_rf, 'time': t1-t0, 'y_pred':y_pred_rf, 'report':report_rf, 'f1':f1_rf}

    # Linear SVM (from-scratch OvR)
    print("Training Linear SVM (OvR) from-scratch...")
    t0 = time.time()
    svm_models, svm_classes = train_ovr_svm(X_train, y_train, C=0.5, lr=0.01, epochs=200, batch_size=64, verbose=False)
    t1 = time.time()
    y_pred_svm = predict_ovr_svm(svm_models, svm_classes, X_test)
    acc_svm = np.mean(y_test == y_pred_svm)
    report_svm, _, _, f1_svm, _ = classification_report_custom(y_test, y_pred_svm, class_names=[k for k,v in sorted(label_map.items(), key=lambda x:x[1])])
    results['LinearSVM_OvR'] = {'acc':acc_svm, 'time': t1-t0, 'y_pred':y_pred_svm, 'report':report_svm, 'f1':f1_svm}

    # Logistic Regression (multiclass softmax) from-scratch
    print("Training Logistic Regression (multiclass softmax, from-scratch)...")
    t0 = time.time()
    logreg = LogisticRegressionCustom(lr=0.02, reg_lambda=0.001, epochs=500, batch_size=64, decay=0.005, random_state=42)
    logreg.fit(X_train, y_train)
    t1 = time.time()
    y_pred_log = logreg.predict(X_test)
    acc_log = np.mean(y_test == y_pred_log)
    report_log, _, _, f1_log, _ = classification_report_custom(y_test, y_pred_log, class_names=[k for k,v in sorted(label_map.items(), key=lambda x:x[1])])
    results['LogisticRegression'] = {'acc':acc_log, 'time': t1-t0, 'y_pred':y_pred_log, 'report':report_log, 'f1':f1_log}

    # Summaries
    print("\nSUMMARY:")
    for name, info in results.items():
        print(f"\n{name}: accuracy={info['acc']:.4f}, time={info['time']:.4f}s")
        print(info['report'])

    # Plots: accuracy and training time
    names = list(results.keys())
    accuracies = [results[n]['acc'] for n in names]
    times = [results[n]['time'] for n in names]

    fig = plt.figure(figsize=(8,4))
    sns.barplot(x=names, y=accuracies)
    plt.ylim(0,1.0)
    plt.title("Test Accuracy")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20)
    acc_path = os.path.join(OUTPUT_DIR, "accuracy.png")
    fig.savefig(acc_path, bbox_inches='tight'); plt.close(fig)

    fig = plt.figure(figsize=(8,4))
    sns.barplot(x=names, y=times)
    plt.title("Training Time (s)")
    plt.ylabel("Seconds")
    plt.xticks(rotation=20)
    time_path = os.path.join(OUTPUT_DIR, "train_time.png")
    fig.savefig(time_path, bbox_inches='tight'); plt.close(fig)

    # Per-class F1
    classes = [k for k,v in sorted(label_map.items(), key=lambda x:x[1])]
    f1_matrix = np.vstack([results[m]['f1'] for m in names])  # (n_models, n_classes)
    x = np.arange(len(classes))
    width = 0.18
    fig = plt.figure(figsize=(10,5))
    for i, row in enumerate(f1_matrix):
        plt.bar(x + i*width, row, width=width, label=names[i])
    plt.xticks(x + width*(len(names)-1)/2, classes)
    plt.ylim(0,1)
    plt.ylabel("F1 score")
    plt.title("Per-class F1 comparison")
    plt.legend()
    f1_path = os.path.join(OUTPUT_DIR, "per_class_f1.png")
    fig.savefig(f1_path, bbox_inches='tight'); plt.close(fig)

    # Confusion matrices
    n = len(names); cols = 2; rows = (n+1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
    axes = axes.flatten()
    for i, name in enumerate(names):
        ax = axes[i]
        cm = np.zeros((len(label_map), len(label_map)), dtype=int)
        pred = results[name]['y_pred']
        for t,p in zip(y_test, pred):
            cm[int(t), int(p)] += 1
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=classes, yticklabels=classes)
        ax.set_title(name)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    fig.savefig(cm_path, bbox_inches='tight'); plt.close(fig)

    # Permutation importance for RF
    print("Permutation feature importance for Random Forest (this may take a while)...")
    fi_df = permutation_feature_importance(rf, X_test, y_test, feature_names, n_repeats=3)
    fi_path = os.path.join(OUTPUT_DIR, "rf_feature_importance.png")
    topk = min(25, len(fi_df))
    fig = plt.figure(figsize=(8,6))
    # seaborn is already imported as `sns` at module level; avoid re-importing locally to prevent scoping issues
    sns.barplot(x='importance', y='feature', data=fi_df.head(topk))
    plt.title("Permutation Feature Importance (Random Forest)")
    fig.savefig(fi_path, bbox_inches='tight'); plt.close(fig)

    # Save artifacts
    artifacts = {
        'results': results,
        'rf_model': rf,
        'svm_models': svm_models,
        'svm_classes': svm_classes,
        'knn_model': knn,
        'decision_tree_model': dt,
        'logistic_model': logreg,
        'feature_names': feature_names,
        'scaler': scaler,
        'label_map': label_map
    }
    with open(os.path.join(ARTIFACTS, "merged_models_artifacts.pkl"), 'wb') as f:
        pickle.dump(artifacts, f)

    # Simple HTML report
    html = f"""
    <html><head><title>Model Comparison Report</title></head><body>
    <h1>Sleep Disorder Models Comparison</h1>
    <h2>Summary</h2>
    <ul>
    """
    for name, info in results.items():
        html += f"<li><b>{name}</b>: accuracy={info['acc']:.4f}, training_time={info['time']:.2f}s</li>"
    html += "</ul>"
    html += f"<h2>Plots</h2>"
    html += f"<img src='{os.path.basename(acc_path)}' width='600'><br>"
    html += f"<img src='{os.path.basename(time_path)}' width='600'><br>"
    html += f"<img src='{os.path.basename(f1_path)}' width='900'><br>"
    html += f"<img src='{os.path.basename(cm_path)}' width='900'><br>"
    html += f"<img src='{os.path.basename(fi_path)}' width='900'><br>"
    html += "</body></html>"

    report_path = os.path.join(OUTPUT_DIR, "report.html")
    with open(report_path, 'w', encoding='utf-8') as fh:
        fh.write(html)
    # copy images into same directory (they are already saved there)
    print(f"Saved report to {report_path}")
    return results, artifacts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train & compare models (from-scratch) for sleep disorder dataset.")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    args = parser.parse_args()
    results, artifacts = run_all_models(args.data)
    print("Done. Outputs in ./outputs and artifacts in ./artifacts")
