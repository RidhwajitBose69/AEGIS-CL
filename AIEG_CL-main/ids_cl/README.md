# IDS-CL: Severity-Aware Continual Learning for Network Intrusion Detection

Track 5 — Continual Learning

## The problem

Network intrusion detection systems must learn to detect new attack types
(ransomware, zero-days) as they emerge — without forgetting how to detect
older attacks (DoS floods, port scans). Naive fine-tuning on new threat data
causes catastrophic forgetting of old threat signatures, creating real
security vulnerabilities.

Standard continual learning methods (EWC, replay) mitigate forgetting but
treat all attack types as equally important. In cybersecurity, this is wrong:
forgetting how to detect a rare privilege-escalation (U2R) attack is
catastrophically worse than forgetting a common DoS flood pattern.

## Our contribution

Two severity-aware modifications to established CL methods:

1. **Severity-Weighted EWC** — Fisher information penalty scaled by threat
   severity per class. Weights critical for detecting U2R/R2L attacks get
   5x/4x the protection of weights critical for Normal traffic.

2. **Priority Replay** — buffer slots allocated proportionally to threat
   severity instead of equally per class. Same total buffer size as standard
   replay (fair comparison), but critical attacks get more rehearsal.

3. **Weighted Security Accuracy (WSA)** — a domain-specific evaluation metric
   that weights each correct prediction by its class's threat severity.
   Standard ACC treats all classes equally; WSA reveals when a model has
   sacrificed critical-attack detection for easy common-traffic accuracy.

## Dataset

**NSL-KDD** — standard cybersecurity benchmark, freely available.
- ~125,000 network connection records
- 41 features (protocol, service, byte counts, error rates, etc.)
- 5 categories: Normal, DoS, Probe, R2L, U2R

Download from: https://www.unb.ca/cic/datasets/nsl.html
Place `KDDTrain+.txt` and `KDDTest+.txt` in `./data/`

If files are absent, the pipeline auto-falls back to synthetic data for testing.

## Setup

```bash
pip install torch pandas scikit-learn matplotlib
```

## Run (one command)

```bash
python run_all.py
```

Produces in `./results/`:
- `results_table.csv` — ACC + forgetting + WSA for all 6 methods
- `accuracy_matrices.txt` — full per-task accuracy matrices
- `comparison_plot.png` — side-by-side ACC vs WSA bar chart

## Experiment matrix

| Method | Role | Origin |
|---|---|---|
| Naive sequential FT | Baseline (lower bound) | Standard |
| Joint training | Baseline (upper bound) | Standard |
| EWC | Model 1 | Established (Kirkpatrick et al., 2017) |
| Replay | Model 2 | Established |
| Severity-Weighted EWC | Model 3 | **Ours** |
| Priority Replay | Model 4 | **Ours** |

## Metrics

- **Average Accuracy (ACC)** — fixed by the track
- **Average Forgetting (BWT)** — fixed by the track
- **Weighted Security Accuracy (WSA)** — ours, domain-specific

## Ablation

The natural ablation is **severity-weighted vs. standard** — same method
(EWC or replay), same buffer size, same hyperparameters, only the
severity weighting differs. This isolates the effect of cost-sensitivity.

## Reproducibility

- `seed=42` fixed everywhere
- One-command run: `python run_all.py`
- Test split used only for final evaluation, never for training/tuning

## Citations

- NSL-KDD: Tavallaee et al., "A detailed analysis of the KDD CUP 99 data set," IEEE CISDA, 2009.
- EWC: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks," PNAS, 2017.
- Experience Replay: Rolnick et al., "Experience Replay for Continual Learning," NeurIPS, 2019.
