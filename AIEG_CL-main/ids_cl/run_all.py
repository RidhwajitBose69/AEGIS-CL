"""
IDS-CL — ONE-COMMAND RUNNER

    python run_all.py

Runs, in order:
  1. Naive sequential            (lower bound, mandatory)
  2. Joint training              (upper bound, mandatory)
  3. Standard EWC                (established baseline)
  4. Standard Replay             (established baseline)
  5. Severity-Weighted EWC       (ours — protects critical attack knowledge more)
  6. Priority Replay             (ours — rehearses high-severity attacks more)

Reports: ACC, average forgetting, AND Weighted Security Accuracy (WSA)
for every method. WSA is our domain-specific metric that penalizes
forgetting critical attacks more than forgetting common ones.

seed=42 fixed throughout.
"""

import csv
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ContinualIDS, set_seed
from data import (build_task_loaders, TASK_SPLITS, CATEGORY_NAMES,
                  SEVERITY_WEIGHTS, NUM_CLASSES, get_input_dim)
from strategies import (train_naive, train_joint, train_ewc, train_replay,
                         train_severity_ewc, train_priority_replay)
from metrics import evaluate, evaluate_weighted, AccuracyMatrix

SEED = 42
RESULTS_DIR = "./results"
EPOCHS_PER_TASK = 20


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)

    print("=" * 60)
    print("IDS-CL: Continual Learning for Network Intrusion Detection")
    print("=" * 60)
    train_loaders, test_loaders, input_dim = build_task_loaders()

    def model_fn():
        return ContinualIDS(input_dim=input_dim, num_classes=NUM_CLASSES)

    # Evaluate WSA across ALL test loaders for a given model
    def compute_wsa(model):
        from torch.utils.data import ConcatDataset, DataLoader
        all_test = ConcatDataset([dl.dataset for dl in test_loaders])
        all_loader = DataLoader(all_test, batch_size=512, shuffle=False)
        device = next(model.parameters()).device
        return evaluate_weighted(model, all_loader, SEVERITY_WEIGHTS, device)

    results = {}  # method -> (ACC, forgetting, WSA)
    matrices = []

    # ---- 1. Naive ----
    print("\n" + "=" * 60 + "\n1. NAIVE SEQUENTIAL FINE-TUNING\n" + "=" * 60)
    set_seed(SEED)
    naive_model, naive_am = train_naive(model_fn, train_loaders, test_loaders, EPOCHS_PER_TASK)
    results["Naive (lower bound)"] = (naive_am.average_accuracy(), naive_am.average_forgetting(), compute_wsa(naive_model))
    matrices.append(("Naive", naive_am))

    # ---- 2. Joint ----
    print("\n" + "=" * 60 + "\n2. JOINT TRAINING\n" + "=" * 60)
    set_seed(SEED)
    joint_model, joint_am = train_joint(model_fn, train_loaders, test_loaders, EPOCHS_PER_TASK)
    results["Joint (upper bound)"] = (joint_am.average_accuracy(), joint_am.average_forgetting(), compute_wsa(joint_model))
    matrices.append(("Joint", joint_am))

    # ---- 3. Standard EWC ----
    print("\n" + "=" * 60 + "\n3. STANDARD EWC\n" + "=" * 60)
    set_seed(SEED)
    ewc_model, ewc_am = train_ewc(model_fn, train_loaders, test_loaders, EPOCHS_PER_TASK)
    results["EWC (standard)"] = (ewc_am.average_accuracy(), ewc_am.average_forgetting(), compute_wsa(ewc_model))
    matrices.append(("EWC", ewc_am))

    # ---- 4. Standard Replay ----
    print("\n" + "=" * 60 + "\n4. STANDARD REPLAY\n" + "=" * 60)
    set_seed(SEED)
    replay_model, replay_am = train_replay(model_fn, train_loaders, test_loaders, EPOCHS_PER_TASK)
    results["Replay (standard)"] = (replay_am.average_accuracy(), replay_am.average_forgetting(), compute_wsa(replay_model))
    matrices.append(("Replay", replay_am))

    # ---- 5. Severity-Weighted EWC (OURS) ----
    print("\n" + "=" * 60 + "\n5. SEVERITY-WEIGHTED EWC (ours)\n" + "=" * 60)
    set_seed(SEED)
    sewc_model, sewc_am = train_severity_ewc(
        model_fn, train_loaders, test_loaders, TASK_SPLITS, SEVERITY_WEIGHTS, EPOCHS_PER_TASK)
    results["Severity-EWC (ours)"] = (sewc_am.average_accuracy(), sewc_am.average_forgetting(), compute_wsa(sewc_model))
    matrices.append(("Severity-EWC", sewc_am))

    # ---- 6. Priority Replay (OURS) ----
    print("\n" + "=" * 60 + "\n6. PRIORITY REPLAY (ours)\n" + "=" * 60)
    set_seed(SEED)
    preplay_model, preplay_am = train_priority_replay(
        model_fn, train_loaders, test_loaders, TASK_SPLITS, SEVERITY_WEIGHTS, EPOCHS_PER_TASK)
    results["Priority-Replay (ours)"] = (preplay_am.average_accuracy(), preplay_am.average_forgetting(), compute_wsa(preplay_model))
    matrices.append(("Priority-Replay", preplay_am))

    # ---- Save results ----
    csv_path = os.path.join(RESULTS_DIR, "results_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Avg Accuracy (ACC)", "Avg Forgetting (BWT)", "Weighted Security Acc (WSA)"])
        for method, (acc, bwt, wsa) in results.items():
            writer.writerow([method, f"{acc:.4f}", f"{bwt:.4f}", f"{wsa:.4f}"])
    print(f"\n[saved] {csv_path}")

    mat_path = os.path.join(RESULTS_DIR, "accuracy_matrices.txt")
    with open(mat_path, "w") as f:
        for name, m in matrices:
            f.write(f"=== {name} ===\n{m.as_table()}\n\n")
    print(f"[saved] {mat_path}")

    # ---- The comparison plot ----
    methods = list(results.keys())
    acc_vals = [results[m][0] for m in methods]
    wsa_vals = [results[m][2] for m in methods]
    short_names = ["Naive", "Joint", "EWC", "Replay", "Sev-EWC\n(ours)", "Pri-Replay\n(ours)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#C9451F', '#1E2761', '#5B6B8C', '#5B6B8C', '#C9820C', '#C9820C']
    ax1.bar(range(len(methods)), acc_vals, color=colors)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(short_names, fontsize=9)
    ax1.set_ylabel("Average Accuracy (ACC)")
    ax1.set_title("Standard Metric: Average Accuracy")
    ax1.set_ylim(0, 1.0)

    ax2.bar(range(len(methods)), wsa_vals, color=colors)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(short_names, fontsize=9)
    ax2.set_ylabel("Weighted Security Accuracy (WSA)")
    ax2.set_title("Our Metric: Severity-Weighted Accuracy")
    ax2.set_ylim(0, 1.0)

    plt.suptitle("Standard CL treats all attacks equally — ours protects critical threats",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "comparison_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[saved] {plot_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':35s} {'ACC':>8s}  {'Forget':>8s}  {'WSA':>8s}")
    print("-" * 65)
    for method, (acc, bwt, wsa) in results.items():
        print(f"{method:35s} {acc:8.4f}  {bwt:8.4f}  {wsa:8.4f}")


if __name__ == "__main__":
    main()
