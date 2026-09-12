"""
IDS-CL — NSL-KDD data loader.

Loads the NSL-KDD dataset and splits it into sequential tasks by attack category.
Includes threat-severity weights — the core domain knowledge that makes our
cost-sensitive CL approach meaningful.

Task structure (class-incremental):
  Task 1: Normal (0) vs DoS (1)        — baseline traffic vs flooding
  Task 2: Add Probe (2)                — surveillance/scanning
  Task 3: Add R2L (3)                  — remote-to-local intrusion
  Task 4: Add U2R (4)                  — privilege escalation (rarest, most critical)

Threat severity weights (domain-informed, not arbitrary):
  Normal:  1.0  (misclassifying normal as attack = false alarm, annoying but not dangerous)
  DoS:     2.0  (service disruption, but usually detected by volume-based rules too)
  Probe:   3.0  (reconnaissance — precursor to real attacks, missing this is bad)
  R2L:     4.0  (actual intrusion — data breach risk)
  U2R:     5.0  (full system compromise — worst-case security outcome)

NSL-KDD download: https://www.unb.ca/cic/datasets/nsl.html
We use KDDTrain+ and KDDTest+ files.
If not available locally, falls back to synthetic data for pipeline testing.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Attack category mapping (NSL-KDD attack names → 5 categories)
ATTACK_CAT = {
    'normal': 'Normal',
    # DoS
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS',
    # Probe
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'snmpgetattack': 'R2L', 'named': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L',
    'sendmail': 'R2L', 'httptunnel': 'R2L', 'worm': 'R2L', 'snmpguess': 'R2L',
    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'xterm': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R',
}

CATEGORY_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
CAT_TO_IDX = {c: i for i, c in enumerate(CATEGORY_NAMES)}
NUM_CLASSES = len(CATEGORY_NAMES)

# Task splits: which class labels appear in each task
TASK_SPLITS = [
    [0, 1],     # Task 1: Normal vs DoS
    [2],        # Task 2: add Probe
    [3],        # Task 3: add R2L
    [4],        # Task 4: add U2R
]
NUM_TASKS = len(TASK_SPLITS)

# Threat severity weights — domain knowledge, not tuned
SEVERITY_WEIGHTS = {
    0: 1.0,   # Normal — false alarm cost
    1: 2.0,   # DoS — service disruption
    2: 3.0,   # Probe — reconnaissance, precursor to real attack
    3: 4.0,   # R2L — actual intrusion, data breach
    4: 5.0,   # U2R — full system compromise
}

NSL_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack', 'difficulty'
]


def _load_nsl_kdd(data_dir="./data"):
    """Load NSL-KDD from local CSV files. Returns (X_train, y_train, X_test, y_test)."""
    train_path = os.path.join(data_dir, "KDDTrain+.txt")
    test_path = os.path.join(data_dir, "KDDTest+.txt")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        return None

    train_df = pd.read_csv(train_path, header=None, names=NSL_COLUMNS)
    test_df = pd.read_csv(test_path, header=None, names=NSL_COLUMNS)

    # Map attack names to categories
    train_df['category'] = train_df['attack'].str.strip().str.lower().map(
        lambda x: CAT_TO_IDX.get(ATTACK_CAT.get(x, 'Normal'), 0))
    test_df['category'] = test_df['attack'].str.strip().str.lower().map(
        lambda x: CAT_TO_IDX.get(ATTACK_CAT.get(x, 'Normal'), 0))

    # Drop non-feature columns
    drop_cols = ['attack', 'difficulty']
    X_train_raw = train_df.drop(columns=drop_cols + ['category'])
    X_test_raw = test_df.drop(columns=drop_cols + ['category'])

    # Encode categorical features (protocol_type, service, flag)
    cat_cols = ['protocol_type', 'service', 'flag']
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train_raw[col], X_test_raw[col]], axis=0).astype(str)
        le.fit(combined)
        X_train_raw[col] = le.transform(X_train_raw[col].astype(str))
        X_test_raw[col] = le.transform(X_test_raw[col].astype(str))
        le_dict[col] = le

    # One-hot encode categorical columns
    X_train_enc = pd.get_dummies(X_train_raw, columns=cat_cols)
    X_test_enc = pd.get_dummies(X_test_raw, columns=cat_cols)

    # Align columns (test may have different one-hot columns)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join='left', axis=1, fill_value=0)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_enc.values.astype(np.float32))
    X_test = scaler.transform(X_test_enc.values.astype(np.float32))

    y_train = train_df['category'].values.astype(np.int64)
    y_test = test_df['category'].values.astype(np.int64)

    return X_train, y_train, X_test, y_test


def _generate_synthetic(seed=42):
    """Synthetic fallback for pipeline testing when NSL-KDD files are absent."""
    rng = np.random.RandomState(seed)
    input_dim = 122
    samples_per_class = {'Normal': 600, 'DoS': 500, 'Probe': 300, 'R2L': 100, 'U2R': 30}

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for cat, n in samples_per_class.items():
        idx = CAT_TO_IDX[cat]
        center = rng.randn(input_dim) * (idx + 1) * 0.3
        X = rng.randn(n, input_dim) * 0.5 + center
        y = np.full(n, idx, dtype=np.int64)

        split = int(n * 0.8)
        X_train_list.append(X[:split])
        y_train_list.append(y[:split])
        X_test_list.append(X[split:])
        y_test_list.append(y[split:])

    return (np.vstack(X_train_list).astype(np.float32), np.concatenate(y_train_list),
            np.vstack(X_test_list).astype(np.float32), np.concatenate(y_test_list))


def load_data(data_dir="./data"):
    result = _load_nsl_kdd(data_dir)
    if result is not None:
        print(f"[data] Loaded real NSL-KDD from {data_dir}")
        return result
    print("[data] NSL-KDD files not found. Using SYNTHETIC fallback for pipeline testing.")
    print("       Download KDDTrain+.txt and KDDTest+.txt from:")
    print("       https://www.unb.ca/cic/datasets/nsl.html")
    print("       Place them in ./data/ and re-run for real results.")
    return _generate_synthetic()


def get_input_dim(data_dir="./data"):
    """Returns actual input dimension (varies with one-hot encoding)."""
    X_train, _, _, _ = load_data(data_dir)
    return X_train.shape[1]


def build_task_loaders(data_dir="./data", batch_size=256):
    """
    Returns (train_loaders, test_loaders) — lists of length NUM_TASKS.
    Each loader contains only the classes assigned to that task.
    """
    X_train, y_train, X_test, y_test = load_data(data_dir)
    input_dim = X_train.shape[1]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loaders, test_loaders = [], []
    for task_classes in TASK_SPLITS:
        # Train
        mask = torch.zeros(len(y_train_t), dtype=torch.bool)
        for c in task_classes:
            mask |= (y_train_t == c)
        train_ds = TensorDataset(X_train_t[mask], y_train_t[mask])
        train_loaders.append(DataLoader(train_ds, batch_size=batch_size, shuffle=True))

        # Test
        mask = torch.zeros(len(y_test_t), dtype=torch.bool)
        for c in task_classes:
            mask |= (y_test_t == c)
        test_ds = TensorDataset(X_test_t[mask], y_test_t[mask])
        test_loaders.append(DataLoader(test_ds, batch_size=batch_size, shuffle=False))

    print(f"[data] Input dim: {input_dim}")
    for t, (tr, te) in enumerate(zip(train_loaders, test_loaders)):
        classes = [CATEGORY_NAMES[c] for c in TASK_SPLITS[t]]
        print(f"  Task {t+1} ({', '.join(classes)}): train={len(tr.dataset)} test={len(te.dataset)}")

    return train_loaders, test_loaders, input_dim


if __name__ == "__main__":
    train_loaders, test_loaders, input_dim = build_task_loaders()
