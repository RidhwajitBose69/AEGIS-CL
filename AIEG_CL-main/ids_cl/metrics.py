"""
IDS-CL — Metrics harness.

Track 5 fixed metrics:
  1. Average Accuracy (ACC) after the final task
  2. Average Forgetting / Backward Transfer (BWT)

Plus our domain-specific metric:
  3. Weighted Security Accuracy (WSA) — accuracy weighted by threat severity
"""

import torch


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_weighted(model, loader, severity_weights, device="cpu"):
    """Weighted Security Accuracy — each correct prediction contributes
    its class's severity weight. Denominator is sum of all weights.
    A model that forgets rare-but-critical U2R attacks gets punished
    more than one that forgets common DoS patterns."""
    model.eval()
    weighted_correct, weighted_total = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        for i in range(y.size(0)):
            w = severity_weights.get(y[i].item(), 1.0)
            weighted_total += w
            if preds[i] == y[i]:
                weighted_correct += w
    return weighted_correct / weighted_total if weighted_total > 0 else 0.0


class AccuracyMatrix:
    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        self.R = [[None] * num_tasks for _ in range(num_tasks)]

    def log(self, task_i: int, after_task_j: int, accuracy: float):
        self.R[task_i][after_task_j] = accuracy

    def log_all_seen_tasks(self, model, test_loaders, after_task_j: int, device="cpu"):
        for i in range(after_task_j + 1):
            acc = evaluate(model, test_loaders[i], device=device)
            self.log(i, after_task_j, acc)

    def average_accuracy(self, after_task_j: int = None) -> float:
        j = self.num_tasks - 1 if after_task_j is None else after_task_j
        vals = [self.R[i][j] for i in range(j + 1) if self.R[i][j] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def average_forgetting(self, final_task: int = None) -> float:
        T = self.num_tasks - 1 if final_task is None else final_task
        forgettings = []
        for i in range(T):
            seen = [self.R[i][k] for k in range(i, T) if self.R[i][k] is not None]
            if not seen or self.R[i][T] is None:
                continue
            forgettings.append(max(seen) - self.R[i][T])
        return sum(forgettings) / len(forgettings) if forgettings else 0.0

    def as_table(self):
        rows = []
        header = "task\\after " + " ".join(f"T{j+1}" for j in range(self.num_tasks))
        rows.append(header)
        for i in range(self.num_tasks):
            row = [f"T{i+1}    "]
            for j in range(self.num_tasks):
                v = self.R[i][j]
                row.append(f"{v:.3f}" if v is not None else "  -  ")
            rows.append(" ".join(row))
        return "\n".join(rows)
