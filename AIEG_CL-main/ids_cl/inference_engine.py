"""
IDS-CL: Inference Engine & Preprocessing Pipeline.

Loads trained model checkpoints, manages feature normalization,
provides sample attack packets, and executes live threat predictions.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import ContinualIDS
from data import CATEGORY_NAMES, SEVERITY_WEIGHTS, NUM_CLASSES

CHECKPOINTS_DIR = "./checkpoints"
PREPROCESSOR_PATH = os.path.join(CHECKPOINTS_DIR, "preprocessor.pkl")

# Curated benchmark packet signatures representing each class
CURATED_SAMPLES = {
    "Normal - Standard Web Traffic": {
        "description": "Benign HTTP session with normal byte transfer and successful handshake.",
        "category": "Normal",
        "ground_truth_class": 0,
        "features": {
            "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
            "src_bytes": 215, "dst_bytes": 4500, "land": 0, "wrong_fragment": 0,
            "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1,
            "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
            "num_file_creations": 0, "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0,
            "is_host_login": 0, "is_guest_login": 0, "count": 1, "srv_count": 1,
            "serror_rate": 0.0, "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
            "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
            "dst_host_count": 30, "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.03,
            "dst_host_srv_diff_host_rate": 0.04, "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
        }
    },
    "DoS - Neptune SYN Flood": {
        "description": "High-volume SYN flood targeting port 80 with SYN errors and zero destination bytes.",
        "category": "DoS",
        "ground_truth_class": 1,
        "features": {
            "duration": 0, "protocol_type": "tcp", "service": "private", "flag": "S0",
            "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0,
            "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 0,
            "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
            "num_file_creations": 0, "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0,
            "is_host_login": 0, "is_guest_login": 0, "count": 220, "srv_count": 12,
            "serror_rate": 1.0, "srv_serror_rate": 1.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
            "same_srv_rate": 0.05, "diff_srv_rate": 0.07, "srv_diff_host_rate": 0.0,
            "dst_host_count": 255, "dst_host_srv_count": 12, "dst_host_same_srv_rate": 0.05,
            "dst_host_diff_srv_rate": 0.07, "dst_host_same_src_port_rate": 0.0,
            "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 1.0,
            "dst_host_srv_serror_rate": 1.0, "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
        }
    },
    "Probe - Portsweep Reconnaissance": {
        "description": "Systematic port sweep scanning multiple ports on a host to identify vulnerabilities.",
        "category": "Probe",
        "ground_truth_class": 2,
        "features": {
            "duration": 0, "protocol_type": "tcp", "service": "private", "flag": "REJ",
            "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0,
            "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 0,
            "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
            "num_file_creations": 0, "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0,
            "is_host_login": 0, "is_guest_login": 0, "count": 110, "srv_count": 2,
            "serror_rate": 0.0, "srv_serror_rate": 0.0, "rerror_rate": 1.0, "srv_rerror_rate": 1.0,
            "same_srv_rate": 0.02, "diff_srv_rate": 0.06, "srv_diff_host_rate": 0.0,
            "dst_host_count": 255, "dst_host_srv_count": 2, "dst_host_same_srv_rate": 0.01,
            "dst_host_diff_srv_rate": 0.65, "dst_host_same_src_port_rate": 1.0,
            "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 1.0, "dst_host_srv_rerror_rate": 1.0
        }
    },
    "R2L - Guess Password / Brute Force": {
        "description": "Multiple failed authentication attempts trying to breach remote Telnet/FTP access.",
        "category": "R2L",
        "ground_truth_class": 3,
        "features": {
            "duration": 1, "protocol_type": "tcp", "service": "telnet", "flag": "SF",
            "src_bytes": 126, "dst_bytes": 179, "land": 0, "wrong_fragment": 0,
            "urgent": 0, "hot": 1, "num_failed_logins": 3, "logged_in": 0,
            "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
            "num_file_creations": 0, "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0,
            "is_host_login": 0, "is_guest_login": 0, "count": 1, "srv_count": 1,
            "serror_rate": 0.0, "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
            "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
            "dst_host_count": 1, "dst_host_srv_count": 1, "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 1.0,
            "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
        }
    },
    "U2R - Buffer Overflow Privilege Escalation": {
        "description": "Local buffer overflow exploit attempting to spawn a root shell and gain superuser rights.",
        "category": "U2R",
        "ground_truth_class": 4,
        "features": {
            "duration": 184, "protocol_type": "tcp", "service": "telnet", "flag": "SF",
            "src_bytes": 1511, "dst_bytes": 2957, "land": 0, "wrong_fragment": 0,
            "urgent": 0, "hot": 3, "num_failed_logins": 0, "logged_in": 1,
            "num_compromised": 2, "root_shell": 1, "su_attempted": 1, "num_root": 3,
            "num_file_creations": 2, "num_shells": 1, "num_access_files": 1, "num_outbound_cmds": 0,
            "is_host_login": 0, "is_guest_login": 0, "count": 1, "srv_count": 1,
            "serror_rate": 0.0, "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
            "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
            "dst_host_count": 1, "dst_host_srv_count": 1, "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 1.0,
            "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
        }
    }
}

TRIAGE_RECOMMENDATIONS = {
    'Normal': {
        'status': 'PASS',
        'action': 'Normal traffic allowed through perimeter firewall. Standard session logging.',
        'severity_color': '#00ff9d'
    },
    'DoS': {
        'status': 'THROTTLE',
        'action': 'Engage automated rate limiting & IP flood mitigation. Apply edge syncookies.',
        'severity_color': '#38bdf8'
    },
    'Probe': {
        'status': 'ALERT & TRACK',
        'action': 'Add scanning source IP to dynamic greylist. Monitor subsequent connection requests.',
        'severity_color': '#ffb700'
    },
    'R2L': {
        'status': 'BLOCK & ISOLATE',
        'action': 'Terminating compromised remote session. Enforce multi-factor auth lockout.',
        'severity_color': '#f97316'
    },
    'U2R': {
        'status': 'CRITICAL INCIDENT',
        'action': 'HOST QUARANTINE: Privilege escalation detected. Revoke root tokens and dump memory forensics.',
        'severity_color': '#ff3366'
    }
}


def load_preprocessor():
    """Load cached preprocessor dictionary."""
    if not os.path.exists(PREPROCESSOR_PATH):
        return None
    with open(PREPROCESSOR_PATH, "rb") as f:
        return pickle.load(f)


def transform_packet_features(raw_dict, preprocessor):
    """
    Transforms a dictionary of raw connection features into normalized model tensor (1, 122).
    """
    scaler = preprocessor['scaler']
    cat_cols = preprocessor['cat_cols']
    le_dict = preprocessor['le_dict']
    expected_cols = preprocessor['columns']

    # Convert to DataFrame
    df = pd.DataFrame([raw_dict])

    # Encode categorical columns
    for col in cat_cols:
        if col in df:
            val = str(df[col].iloc[0])
            le = le_dict[col]
            if val in le.classes_:
                df[col] = le.transform([val])[0]
            else:
                df[col] = 0

    # One-hot encode categorical features
    df_enc = pd.get_dummies(df, columns=cat_cols)

    # Reindex/align to match training columns
    for col in expected_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0.0
    df_enc = df_enc[expected_cols]

    # Standard scale
    scaled_arr = scaler.transform(df_enc.values.astype(np.float32))
    return torch.tensor(scaled_arr, dtype=torch.float32)


def load_model_checkpoint(method_key, input_dim=122):
    """Loads a single model checkpoint by key."""
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{method_key}.pt")
    if not os.path.exists(ckpt_path):
        return None, None
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ContinualIDS(input_dim=ckpt.get('input_dim', input_dim), num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def load_all_models(input_dim=122):
    """Loads all 6 model checkpoints into a dict."""
    methods = {
        'naive': 'Naive (lower bound)',
        'joint': 'Joint (upper bound)',
        'ewc': 'EWC (standard)',
        'replay': 'Replay (standard)',
        'severity_ewc': 'Severity-EWC (ours)',
        'priority_replay': 'Priority-Replay (ours)'
    }
    models = {}
    metadata = {}
    for key, display_name in methods.items():
        m, ckpt = load_model_checkpoint(key, input_dim)
        if m is not None:
            models[display_name] = m
            metadata[display_name] = ckpt
    return models, metadata


@torch.no_grad()
def run_threat_inference(model, tensor_input):
    """
    Evaluates model on tensor_input.
    Returns (pred_class_idx, pred_class_name, probabilities_list, latent_feats)
    """
    model.eval()
    logits, feats = model(tensor_input, return_features=True)
    probs = F.softmax(logits, dim=1).squeeze(0).numpy()
    pred_idx = int(np.argmax(probs))
    pred_name = CATEGORY_NAMES[pred_idx]
    return pred_idx, pred_name, probs.tolist(), feats.squeeze(0).numpy()
