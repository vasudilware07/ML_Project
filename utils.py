# utils.py
import re
import pandas as pd
import numpy as np
from collections import OrderedDict

def to_snake_case(name: str) -> str:
    name = str(name)
    name = re.sub(r'[-.\s]+', '_', name)
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    name = re.sub(r'__+', '_', name)
    return name.lower().strip('_')

class StandardScalerCustom:
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self, X):
        X = X.astype(float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self
    def transform(self, X):
        X = X.astype(float)
        return (X - self.mean) / self.std
    def fit_transform(self, X):
        return self.fit(X).transform(X)

def load_and_preprocess(csv_path, target_col='sleep_disorder', drop_id_cols=None):
    df = pd.read_csv(csv_path)
    df.columns = [to_snake_case(c) for c in df.columns]
    target_col = to_snake_case(target_col)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {df.columns.tolist()}")
    df[target_col] = df[target_col].astype(str).str.strip()
    df[target_col] = df[target_col].replace({
        r'(?i)sleep\s*apnoea': 'Sleep Apnea',
        r'(?i)sleep[\-_ ]?apnea': 'Sleep Apnea',
        r'(?i)insomnia': 'Insomnia',
        r'(?i)none': 'None',
        r'(?i)nan': 'None'
    }, regex=True)
    label_map = {'None':0, 'Insomnia':1, 'Sleep Apnea':2}
    df = df[df[target_col].isin(label_map.keys())].copy()
    df['target'] = df[target_col].map(label_map)
    if drop_id_cols is None:
        drop_id_cols = ['person_id', 'id', 'patient_id', 'person id', 'personid']
    for c in drop_id_cols:
        if c in df.columns: df.drop(columns=[c], inplace=True)
    X_raw = df.drop(columns=[target_col, 'target'])
    y = df['target'].values.astype(int)
    X_enc = pd.get_dummies(X_raw, drop_first=False)
    feature_names = X_enc.columns.tolist()
    return X_enc, y, feature_names, X_raw, label_map

def classification_report_custom(y_true, y_pred, class_names=None):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    report = "              precision    recall  f1-score   support\n\n"
    precisions, recalls, f1s, supports = [], [], [], []
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp+fp) if (tp+fp)>0 else 0
        recall = tp / (tp+fn) if (tp+fn)>0 else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
        support = np.sum(y_true == cls)
        name = class_names[int(cls)] if class_names is not None else str(int(cls))
        report += f"{name:>12}       {precision:.2f}      {recall:.2f}      {f1:.2f}        {support}\n"
        precisions.append(precision); recalls.append(recall); f1s.append(f1); supports.append(support)
    report += f"\n    accuracy                           {np.mean(y_true==y_pred):.2f}        {len(y_true)}\n"
    report += f"   macro avg       {np.mean(precisions):.2f}      {np.mean(recalls):.2f}      {np.mean(f1s):.2f}        {len(y_true)}\n"
    return report, np.array(precisions), np.array(recalls), np.array(f1s), np.array(supports)
