"""
IDS-CL — Training strategies.

Standard methods (established):
  1. train_naive          — lower bound (mandatory baseline)
  2. train_joint          — upper bound (mandatory baseline)
  3. train_ewc            — standard EWC (Kirkpatrick et al., 2017)
  4. train_replay          — standard replay (random buffer)

Our contribution (severity-aware):
  5. train_severity_ewc    — EWC with Fisher penalties scaled by threat severity
  6. train_priority_replay — replay with buffer slots allocated by threat severity

The key insight: in cybersecurity, misclassification costs are asymmetric.
Forgetting how to detect U2R (privilege escalation) is far worse than
forgetting some DoS flood patterns. Standard CL treats all classes equally.
Our methods don't.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from metrics import AccuracyMatrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _train_one_epoch(model, loader, optimizer, criterion, extra_loss_fn=None):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if extra_loss_fn is not None:
            loss = loss + extra_loss_fn(model)
        loss.backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# 1. NAIVE SEQUENTIAL (lower bound)
# ---------------------------------------------------------------------------
def train_naive(model_fn, train_loaders, test_loaders, epochs_per_task=10, lr=1e-3):
    model = model_fn().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    num_tasks = len(train_loaders)
    am = AccuracyMatrix(num_tasks)

    for t in range(num_tasks):
        for _ in range(epochs_per_task):
            _train_one_epoch(model, train_loaders[t], optimizer, criterion)
        am.log_all_seen_tasks(model, test_loaders, t, DEVICE)
        print(f"[naive] task {t+1}/{num_tasks} ACC={am.average_accuracy(t):.3f}")

    return model, am


# ---------------------------------------------------------------------------
# 2. JOINT TRAINING (upper bound)
# ---------------------------------------------------------------------------
def train_joint(model_fn, train_loaders, test_loaders, epochs=10, lr=1e-3, batch_size=256):
    model = model_fn().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    num_tasks = len(train_loaders)
    am = AccuracyMatrix(num_tasks)

    all_train = ConcatDataset([dl.dataset for dl in train_loaders])
    joint_loader = DataLoader(all_train, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        _train_one_epoch(model, joint_loader, optimizer, criterion)

    am.log_all_seen_tasks(model, test_loaders, num_tasks - 1, DEVICE)
    for i in range(num_tasks):
        final_acc = am.R[i][num_tasks - 1]
        for j in range(i, num_tasks):
            if am.R[i][j] is None:
                am.R[i][j] = final_acc

    print(f"[joint] ACC={am.average_accuracy():.3f}")
    return model, am


# ---------------------------------------------------------------------------
# 3. STANDARD EWC (Kirkpatrick et al., 2017) — established baseline
# ---------------------------------------------------------------------------
class EWC:
    def __init__(self, model, lam=400.0):
        self.model = model
        self.lam = lam
        self.fisher = {}
        self.opt_params = {}

    def _compute_fisher(self, loader, criterion, n_batches=30):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        count = 0
        for i, (x, y) in enumerate(loader):
            if i >= n_batches:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            self.model.zero_grad()
            loss = criterion(self.model(x), y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            count += 1
        for n in fisher:
            fisher[n] /= max(count, 1)
        return fisher

    def register_task(self, loader, criterion):
        new_fisher = self._compute_fisher(loader, criterion)
        new_params = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        if not self.fisher:
            self.fisher = new_fisher
            self.opt_params = new_params
        else:
            for n in self.fisher:
                self.fisher[n] += new_fisher[n]
            self.opt_params = new_params

    def penalty(self, model):
        if not self.fisher:
            return torch.tensor(0.0, device=DEVICE)
        loss = torch.tensor(0.0, device=DEVICE)
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.opt_params[n]) ** 2).sum()
        return self.lam * loss


def train_ewc(model_fn, train_loaders, test_loaders, epochs_per_task=10, lr=1e-3, ewc_lambda=400.0):
    model = model_fn().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    num_tasks = len(train_loaders)
    am = AccuracyMatrix(num_tasks)
    ewc = EWC(model, lam=ewc_lambda)

    for t in range(num_tasks):
        for _ in range(epochs_per_task):
            _train_one_epoch(model, train_loaders[t], optimizer, criterion,
                             extra_loss_fn=lambda m: ewc.penalty(m))
        ewc.register_task(train_loaders[t], criterion)
        am.log_all_seen_tasks(model, test_loaders, t, DEVICE)
        print(f"[ewc] task {t+1}/{num_tasks} ACC={am.average_accuracy(t):.3f}")

    return model, am


# ---------------------------------------------------------------------------
# 4. STANDARD REPLAY (random buffer) — established baseline
# ---------------------------------------------------------------------------
def train_replay(model_fn, train_loaders, test_loaders, epochs_per_task=10, lr=1e-3,
                 buffer_per_class=50, batch_size=256):
    model = model_fn().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    num_tasks = len(train_loaders)
    am = AccuracyMatrix(num_tasks)
    buffer = []

    for t in range(num_tasks):
        task_ds = train_loaders[t].dataset
        if buffer:
            buf_x = torch.cat([b[0] for b in buffer])
            buf_y = torch.cat([b[1] for b in buffer])
            combined = ConcatDataset([task_ds, TensorDataset(buf_x, buf_y)])
        else:
            combined = task_ds
        loader = DataLoader(combined, batch_size=batch_size, shuffle=True)

        for _ in range(epochs_per_task):
            _train_one_epoch(model, loader, optimizer, criterion)

        # Random buffer update — EQUAL slots per class (standard approach)
        xs, ys = [], []
        for x, y in DataLoader(task_ds, batch_size=512):
            xs.append(x); ys.append(y)
        all_x, all_y = torch.cat(xs), torch.cat(ys)
        for c in all_y.unique():
            idx = (all_y == c).nonzero(as_tuple=True)[0]
            idx = idx[torch.randperm(len(idx))[:buffer_per_class]]
            buffer.append((all_x[idx], all_y[idx]))

        am.log_all_seen_tasks(model, test_loaders, t, DEVICE)
        buf_size = sum(len(b[1]) for b in buffer)
        print(f"[replay] task {t+1}/{num_tasks} ACC={am.average_accuracy(t):.3f} buf={buf_size}")

    return model, am


# ===========================================================================
# 5. SEVERITY-WEIGHTED EWC (our contribution)
#
# Key idea: multiply Fisher penalty by the threat-severity weight of the
# classes that each parameter was important for. Weights that were critical
# for detecting rare, high-severity attacks (U2R, R2L) get a LARGER penalty
# multiplier than weights critical for common attacks (DoS).
#
# Implementation: after each task, compute Fisher per-class (filtering
# batches by class), then combine with severity-weighted sum instead of
# uniform sum. The penalty function is identical to standard EWC after that.
# ===========================================================================
class SeverityEWC:
    def __init__(self, model, severity_weights, lam=400.0):
        self.model = model
        self.severity_weights = severity_weights
        self.lam = lam
        self.fisher = {}
        self.opt_params = {}

    def _compute_class_fisher(self, loader, criterion, class_label, n_batches=30):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        count = 0
        for i, (x, y) in enumerate(loader):
            mask = (y == class_label)
            if mask.sum() == 0:
                continue
            if count >= n_batches:
                break
            x_c, y_c = x[mask].to(DEVICE), y[mask].to(DEVICE)
            self.model.zero_grad()
            loss = criterion(self.model(x_c), y_c)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            count += 1
        for n in fisher:
            fisher[n] /= max(count, 1)
        return fisher

    def register_task(self, loader, criterion, task_classes):
        """Compute per-class Fisher weighted by severity, then accumulate."""
        weighted_fisher = {n: torch.zeros_like(p)
                           for n, p in self.model.named_parameters() if p.requires_grad}

        for c in task_classes:
            w = self.severity_weights.get(c, 1.0)
            class_fisher = self._compute_class_fisher(loader, criterion, c)
            for n in weighted_fisher:
                weighted_fisher[n] += w * class_fisher[n]

        new_params = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        if not self.fisher:
            self.fisher = weighted_fisher
            self.opt_params = new_params
        else:
            for n in self.fisher:
                self.fisher[n] += weighted_fisher[n]
            self.opt_params = new_params

    def penalty(self, model):
        if not self.fisher:
            return torch.tensor(0.0, device=DEVICE)
        loss = torch.tensor(0.0, device=DEVICE)
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.opt_params[n]) ** 2).sum()
        return self.lam * loss


def train_severity_ewc(model_fn, train_loaders, test_loaders, task_splits,
                        severity_weights, epochs_per_task=10, lr=1e-3, ewc_lambda=400.0):
    model = model_fn().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    num_tasks = len(train_loaders)
    am = AccuracyMatrix(num_tasks)
    sewc = SeverityEWC(model, severity_weights, lam=ewc_lambda)

    for t in range(num_tasks):
        for _ in range(epochs_per_task):
            _train_one_epoch(model, train_loaders[t], optimizer, criterion,
                             extra_loss_fn=lambda m: sewc.penalty(m))
        sewc.register_task(train_loaders[t], criterion, task_splits[t])
        am.log_all_seen_tasks(model, test_loaders, t, DEVICE)
        print(f"[severity-ewc] task {t+1}/{num_tasks} ACC={am.average_accuracy(t):.3f}")

    return model, am


# ===========================================================================
# 6. PRIORITY REPLAY (our contribution)
#
# Key idea: instead of allocating EQUAL buffer slots per class (standard
# replay), allocate MORE slots to high-severity attack types. U2R gets 5x
# the buffer slots of Normal. This means the model rehearses critical
# attack patterns more frequently, directly protecting high-value knowledge.
#
# Total buffer size stays the same as standard replay (fair comparison).
# ===========================================================================
def train_priority_replay(model_fn, train_loaders, test_loaders, task_splits,
                           severity_weights, epochs_per_task=10, lr=1e-3,
                           total_buffer_per_task=100, batch_size=256):
    model = model_fn().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    num_tasks = len(train_loaders)
    am = AccuracyMatrix(num_tasks)
    buffer = []

    for t in range(num_tasks):
        task_ds = train_loaders[t].dataset
        if buffer:
            buf_x = torch.cat([b[0] for b in buffer])
            buf_y = torch.cat([b[1] for b in buffer])
            combined = ConcatDataset([task_ds, TensorDataset(buf_x, buf_y)])
        else:
            combined = task_ds
        loader = DataLoader(combined, batch_size=batch_size, shuffle=True)

        for _ in range(epochs_per_task):
            _train_one_epoch(model, loader, optimizer, criterion)

        # Priority buffer: allocate slots proportional to severity weight
        xs, ys = [], []
        for x, y in DataLoader(task_ds, batch_size=512):
            xs.append(x); ys.append(y)
        all_x, all_y = torch.cat(xs), torch.cat(ys)

        classes_in_task = all_y.unique().tolist()
        total_weight = sum(severity_weights.get(c, 1.0) for c in classes_in_task)

        for c in classes_in_task:
            w = severity_weights.get(c, 1.0)
            # Proportional allocation: more severe → more buffer slots
            n_samples = max(1, int(total_buffer_per_task * w / total_weight))
            idx = (all_y == c).nonzero(as_tuple=True)[0]
            n_samples = min(n_samples, len(idx))
            idx = idx[torch.randperm(len(idx))[:n_samples]]
            buffer.append((all_x[idx], all_y[idx]))

        am.log_all_seen_tasks(model, test_loaders, t, DEVICE)
        buf_size = sum(len(b[1]) for b in buffer)
        print(f"[priority-replay] task {t+1}/{num_tasks} ACC={am.average_accuracy(t):.3f} buf={buf_size}")

    return model, am
