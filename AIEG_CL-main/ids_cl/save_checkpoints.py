"""
IDS-CL: Model Checkpoint Generator & Preprocessor Serializer.

Trains the 6 CL strategies (Naive, Joint, EWC, Replay, Severity-EWC, Priority-Replay)
and saves PyTorch state dicts and feature preprocessor to ./checkpoints/
so the Streamlit dashboard can perform instant real-time inference.
"""

import os
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from model import ContinualIDS, set_seed
from data import (ATTACK_CAT, CATEGORY_NAMES, CAT_TO_IDX, TASK_SPLITS,
                  SEVERITY_WEIGHTS, NUM_CLASSES, NSL_COLUMNS, build_task_loaders)
from strategies import (train_naive, train_joint, train_ewc, train_replay,
                         train_severity_ewc, train_priority_replay)

CHECKPOINTS_DIR = "./checkpoints"
DATA_DIR = "./data"
SEED = 42
EPOCHS = 5  # Fast yet effective convergence for interactive demo


def build_and_save_preprocessor():
    """Fit and save preprocessor (scaler, categorical encoders, column template)."""
    train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
    test_path = os.path.join(DATA_DIR, "KDDTest+.txt")

    train_df = pd.read_csv(train_path, header=None, names=NSL_COLUMNS)
    test_df = pd.read_csv(test_path, header=None, names=NSL_COLUMNS)

    train_df['category'] = train_df['attack'].str.strip().str.lower().map(
        lambda x: CAT_TO_IDX.get(ATTACK_CAT.get(x, 'Normal'), 0))
    test_df['category'] = test_df['attack'].str.strip().str.lower().map(
        lambda x: CAT_TO_IDX.get(ATTACK_CAT.get(x, 'Normal'), 0))

    drop_cols = ['attack', 'difficulty', 'category']
    X_train_raw = train_df.drop(columns=drop_cols)
    X_test_raw = test_df.drop(columns=drop_cols)

    cat_cols = ['protocol_type', 'service', 'flag']
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train_raw[col], X_test_raw[col]], axis=0).astype(str)
        le.fit(combined)
        X_train_raw[col] = le.transform(X_train_raw[col].astype(str))
        X_test_raw[col] = le.transform(X_test_raw[col].astype(str))
        le_dict[col] = le

    X_train_enc = pd.get_dummies(X_train_raw, columns=cat_cols)
    X_test_enc = pd.get_dummies(X_test_raw, columns=cat_cols)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join='left', axis=1, fill_value=0)

    scaler = StandardScaler()
    scaler.fit(X_train_enc.values.astype(np.float32))

    # Also collect representative sample records from test set for each category
    sample_records = {}
    for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
        matches = test_df[test_df['category'] == cat_idx]
        if not matches.empty:
            sample_row = matches.iloc[0].to_dict()
            sample_records[cat_name] = sample_row

    preprocessor_data = {
        'scaler': scaler,
        'cat_cols': cat_cols,
        'le_dict': le_dict,
        'columns': list(X_train_enc.columns),
        'input_dim': len(X_train_enc.columns),
        'raw_feature_cols': [c for c in NSL_COLUMNS if c not in ['attack', 'difficulty']],
        'sample_records': sample_records
    }

    prep_path = os.path.join(CHECKPOINTS_DIR, "preprocessor.pkl")
    with open(prep_path, "wb") as f:
        pickle.dump(preprocessor_data, f)
    print(f"[preprocessor] Saved preprocessor to {prep_path}")
    return preprocessor_data


def train_and_save_all():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    prep_data = build_and_save_preprocessor()
    input_dim = prep_data['input_dim']

    train_loaders, test_loaders, _ = build_task_loaders(DATA_DIR)

    def model_fn():
        return ContinualIDS(input_dim=input_dim, num_classes=NUM_CLASSES)

    strategies = [
        ("naive", lambda: train_naive(model_fn, train_loaders, test_loaders, epochs_per_task=EPOCHS)),
        ("joint", lambda: train_joint(model_fn, train_loaders, test_loaders, epochs=EPOCHS)),
        ("ewc", lambda: train_ewc(model_fn, train_loaders, test_loaders, epochs_per_task=EPOCHS)),
        ("replay", lambda: train_replay(model_fn, train_loaders, test_loaders, epochs_per_task=EPOCHS)),
        ("severity_ewc", lambda: train_severity_ewc(model_fn, train_loaders, test_loaders, TASK_SPLITS, SEVERITY_WEIGHTS, epochs_per_task=EPOCHS)),
        ("priority_replay", lambda: train_priority_replay(model_fn, train_loaders, test_loaders, TASK_SPLITS, SEVERITY_WEIGHTS, epochs_per_task=EPOCHS)),
    ]

    for name, train_call in strategies:
        print(f"\n>>> Training and checkpointing: {name} (epochs={EPOCHS})")
        set_seed(SEED)
        model, am = train_call()
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{name}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': input_dim,
            'num_classes': NUM_CLASSES,
            'accuracy_matrix': am.R,
            'avg_acc': am.average_accuracy(),
            'avg_bwt': am.average_forgetting(),
        }, ckpt_path)
        print(f"[saved] Checkpoint saved: {ckpt_path}")

    print("\n[done] All 6 model checkpoints generated successfully!")


if __name__ == "__main__":
    train_and_save_all()
