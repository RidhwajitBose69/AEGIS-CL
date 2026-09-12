# 🛡️ AEGIS-CL: Severity-Aware Continual Learning for Network Intrusion Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63%2B-FF4B4B.svg)](https://streamlit.io/)
[![Benchmark](https://img.shields.io/badge/Benchmark-NSL--KDD-green.svg)](https://www.unb.ca/cic/datasets/nsl.html)
[![Track](https://img.shields.io/badge/Track-Continual%20Learning-purple.svg)]()

**AEGIS-CL** is a research framework and enterprise-grade Security Operations Center (SOC) dashboard designed to solve **Catastrophic Forgetting** in Network Intrusion Detection Systems (NIDS).

Standard continual learning algorithms treat all classes uniformly. In cybersecurity, this assumption fails catastrophically: forgetting a rare **User-to-Root (U2R)** privilege escalation exposes an organization to complete infrastructure compromise, whereas forgetting a common **DoS** signature merely causes transient disruption. **AEGIS-CL** introduces domain-informed severity weighting into continual learning architectures to guarantee high retention on mission-critical attack vectors.

---

## 📌 Key Innovations

1. **Priority Replay (Ours)**: Rehearsal buffer slots are allocated proportionally to threat severity instead of equally. Critical attacks (U2R, R2L) receive up to **5x more rehearsal slots** within the exact same total memory buffer size as standard replay.
2. **Severity-Weighted EWC (Ours)**: Regularizes network parameters by scaling the diagonal Fisher Information penalty by class severity weights ($F_{\text{weighted}} = \sum_c w_c \cdot F_c(\theta)$), imposing stiffer retention penalties on weights vital for critical threat detection.
3. **Weighted Security Accuracy (WSA — Ours)**: A domain-informed evaluation metric that weights each correct classification by its threat severity ($w_{\text{U2R}}=5.0, w_{\text{R2L}}=4.0, w_{\text{Probe}}=3.0, w_{\text{DoS}}=2.0, w_{\text{Normal}}=1.0$). Exposes when naive models inflate accuracy on benign traffic while silently forgetting dangerous zero-days.

---

## 🚀 Quick Start: How to Run Everything

### Prerequisites & Environment Setup

Clone or navigate to the repository and activate your Python environment:

```bash
cd /home/prajwal/Downloads/ids_cl
```

Install the required dependencies:
```bash
pip install torch pandas scikit-learn matplotlib streamlit plotly altair
```

*(If using the configured environment on this machine):*
```bash
source /home/prajwal/Traffic_Baseine/traffic_baseline/dl/bin/activate
```

---

### Option 1: Launch the Interactive SOC Streamlit Dashboard (Recommended)

Launch the full interactive SOC Cockpit with a single command:

```bash
./run_dashboard.sh
```

Alternatively, run directly via Streamlit:
```bash
streamlit run app.py --server.port 8501
```

Once running, open your browser and navigate to:
👉 **[http://localhost:8501](http://localhost:8501)**

#### 🎛️ What You Can Do in the Dashboard:
* **Tab 1: 🌐 Executive SOC Intelligence & Leaderboard**
  * Real-time KPI tiles showing Top Model (`Priority-Replay`), WSA (`69.0%`), and Forgetting Reduction (`86.3%`).
  * Threat Severity Taxonomy cards (Normal, DoS, Probe, R2L, U2R).
  * Interactive Benchmark Leaderboard & Grouped Performance Charts.
  * **The Security Premium (WSA vs. ACC Delta)** divergence chart.
  * 5-axis Multi-Dimensional Defense Radar polygon.

* **Tab 2: 🔄 Continual Learning Dynamics ($R_{i,j}$)**
  * Interactive Accuracy Matrix heatmaps for all 6 methods.
  * Step-by-Step Task Timeline Slider ($T_1 \to T_4$) to observe model retention evolution.
  * Catastrophic Forgetting Decay Trajectory curves.

* **Tab 3: ⚖️ Severity-Aware Innovations & Ablation**
  * Visual buffer allocation comparison (Uniform 20% vs. Severity-proportional Priority Replay).
  * Fisher Information penalty formulation for Severity-EWC.
  * **Financial Breach Loss Simulator**: Interactive cost model calculating financial risk across defense strategies.

* **Tab 4: 🎯 Live Packet Threat Inspector & Real-Time Inference Playground**
  * Select curated real-world attack signatures (Normal Web, Neptune SYN Flood, Portsweep, Password Brute Force, Buffer Overflow U2R).
  * Or craft custom packets with interactive sliders (protocol, service, flags, bytes, failed logins, root shell indicators).
  * Live side-by-side diagnosis comparing **Naive**, **Standard Replay**, and **Priority-Replay**.
  * Class confidence probability breakdown and automated SOC containment playbooks.

* **Tab 5: 🧪 Continual Learning Experiment Studio**
  * Customize hyperparameters (epochs, learning rate, buffer size, custom severity weights).
  * Trigger live class-incremental training sessions with real-time visual progress.

* **Tab 6: 📊 Dataset & Latent Space Explorer**
  * NSL-KDD class distribution analytics highlighting the extreme 1,200:1 imbalance (52 U2R samples vs. 67,343 Normal samples).
  * 2D feature projection scatter plot from `MLPBackbone` 64-dimensional latent embeddings.

---

### Option 2: Train & Save Checkpoints for Fast Inference

To re-train and serialize the 6 model checkpoints into `./checkpoints/` (used by the live inference playground):

```bash
python save_checkpoints.py
```

This generates:
* `./checkpoints/naive.pt`
* `./checkpoints/joint.pt`
* `./checkpoints/ewc.pt`
* `./checkpoints/replay.pt`
* `./checkpoints/severity_ewc.pt`
* `./checkpoints/priority_replay.pt`
* `./checkpoints/preprocessor.pkl`

---

### Option 3: Run Full Benchmark Pipeline (Headless / Terminal)

To run the complete 20-epoch research experiment across all 6 methods and generate evaluation plots and tables:

```bash
python run_all.py
```

This generates in `./results/`:
* `results_table.csv` — Accuracy (ACC), Backward Transfer (BWT), and WSA.
* `accuracy_matrices.txt` — Full $R_{i,j}$ matrices for all 6 methods.
* `comparison_plot.png` — Publication-ready side-by-side ACC vs. WSA bar charts.

---

## 📊 Benchmark Results

Evaluated on the full **NSL-KDD** benchmark (seed=42 fixed throughout):

| Strategy | Role | Avg Accuracy (ACC) | Avg Forgetting (BWT) | Weighted Security Acc (WSA) |
| :--- | :--- | :---: | :---: | :---: |
| **Naive Sequential FT** | Baseline (Lower Bound) | 25.00% | 63.94% | 26.38% |
| **Joint Training** | Baseline (Upper Bound) | 52.25% | 0.00% | 64.59% |
| **Standard EWC** | Established Baseline | 24.91% | 64.16% | 26.29% |
| **Standard Replay** | Established Baseline | 51.35% | 9.30% | 66.99% |
| **Severity-Weighted EWC** | **Ours** | 26.44% | 56.30% | 27.27% |
| **Priority Replay** | **Ours** | **51.65%** | **8.78%** | **68.97%** |

### Key Takeaways:
- **Priority Replay** achieves the highest **WSA (68.97%)**, outperforming standard replay (+1.98%) and even the Joint Training upper bound (+4.38%) on high-severity attack retention.
- Catastrophic forgetting is reduced by **86.3%** compared to naive sequential learning (BWT drops from 63.94% to 8.78%).

---

## 📂 Project Structure

```text
ids_cl/
├── app.py                  # Streamlit SOC Cockpit application
├── run_dashboard.sh        # One-command dashboard launcher script
├── utils_ui.py             # UI design system, CSS styling, and Plotly visualizers
├── inference_engine.py     # Live packet inspection & inference pipeline
├── save_checkpoints.py     # Checkpoint training and serializer
├── model.py                # ContinualIDS & MLPBackbone architecture (122 -> 256 -> 128 -> 64 -> 5)
├── data.py                 # NSL-KDD dataset loader, one-hot encoding, and task splitting
├── strategies.py           # CL implementations (Naive, Joint, EWC, Replay, Sev-EWC, Priority-Replay)
├── metrics.py              # Accuracy Matrix harness, BWT, and Weighted Security Accuracy (WSA)
├── run_all.py              # One-command headless experiment runner
├── checkpoints/            # Serialized PyTorch models (.pt) & preprocessor (.pkl)
│   ├── naive.pt
│   ├── joint.pt
│   ├── ewc.pt
│   ├── replay.pt
│   ├── severity_ewc.pt
│   ├── priority_replay.pt
│   └── preprocessor.pkl
├── data/                   # NSL-KDD dataset files
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
└── results/                # Output benchmark tables, matrices, and plots
    ├── results_table.csv
    ├── accuracy_matrices.txt
    └── comparison_plot.png
```

---

## 🗄️ Dataset: NSL-KDD

* **Total Records**: ~148,517 network connection flows.
* **Raw Features**: 41 network traffic attributes (protocol, service, flags, duration, byte counts, error rates, etc.).
* **Processed Dimension**: 122 one-hot encoded and standardized features.
* **Class Incremental Task Splits**:
  * **Task 1**: Normal (0) vs. DoS (1) — *113,270 train / 17,169 test*
  * **Task 2**: Add Probe (2) — *11,656 train / 2,421 test*
  * **Task 3**: Add R2L (3) — *995 train / 2,887 test*
  * **Task 4**: Add U2R (4) — *52 train / 67 test*

> **Synthetic Fallback**: If `KDDTrain+.txt` and `KDDTest+.txt` are not present in `./data/`, `data.py` automatically generates a synthetic multi-modal dataset so the pipeline and dashboard continue running without crashing.

---

## 🛠️ Troubleshooting & FAQ

* **Port 8501 is already in use:**
  Run the dashboard on an alternative port:
  ```bash
  streamlit run app.py --server.port 8502
  ```

* **To run in background mode on a remote server:**
  ```bash
  nohup ./run_dashboard.sh > dashboard.log 2>&1 &
  ```

* **Missing module errors:**
  Ensure you are using the correct Python interpreter:
  ```bash
  /home/prajwal/Traffic_Baseine/traffic_baseline/dl/bin/python3 app.py
  ```

---

## 📜 Citations & References

1. **NSL-KDD Dataset**: M. Tavallaee, E. Bagheri, W. Lu, and A. Ghorbani, *"A detailed analysis of the KDD CUP 99 data set,"* IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), 2009.
2. **Elastic Weight Consolidation (EWC)**: J. Kirkpatrick et al., *"Overcoming catastrophic forgetting in neural networks,"* Proceedings of the National Academy of Sciences (PNAS), 2017.
3. **Experience Replay**: D. Rolnick et al., *"Experience Replay for Continual Learning,"* NeurIPS, 2019.
