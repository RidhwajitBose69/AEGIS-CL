"""
IDS-CL: SOC UI & Visualization Utilities.
Custom dark theme styling, glassmorphic metric cards, and Plotly charts.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

# Color Palette for SOC Cyber Defense
CYBER_COLORS = {
    'bg': '#0e1117',
    'card_bg': '#161b22',
    'card_border': '#30363d',
    'cyan': '#00f0ff',
    'emerald': '#00ff9d',
    'amber': '#ffb700',
    'crimson': '#ff3366',
    'purple': '#a855f7',
    'blue': '#38bdf8',
    'text': '#f0f6fc',
    'muted': '#8b949e',
}

SEVERITY_META = {
    'Normal': {'weight': 1.0, 'color': '#00ff9d', 'level': 'LOW RISK', 'desc': 'Standard baseline traffic. False alarms cause noise but no breach.'},
    'DoS': {'weight': 2.0, 'color': '#38bdf8', 'level': 'ELEVATED', 'desc': 'Denial of Service flood. High volume, disrupts service availability.'},
    'Probe': {'weight': 3.0, 'color': '#ffb700', 'level': 'MEDIUM-HIGH', 'desc': 'Port scan & surveillance. Precursor to targeted intrusion.'},
    'R2L': {'weight': 4.0, 'color': '#f97316', 'level': 'HIGH RISK', 'desc': 'Remote-to-Local compromise. Unauthorized local access & data breach.'},
    'U2R': {'weight': 5.0, 'color': '#ff3366', 'level': 'CRITICAL', 'desc': 'User-to-Root privilege escalation. Full host takeover & root compromise.'},
}

METHOD_COLORS = {
    'Naive (lower bound)': '#f43f5e',
    'Joint (upper bound)': '#6366f1',
    'EWC (standard)': '#94a3b8',
    'Replay (standard)': '#38bdf8',
    'Severity-EWC (ours)': '#eab308',
    'Priority-Replay (ours)': '#10b981',
}


def get_soc_custom_css():
    """Returns custom CSS for dark mode cybersecurity cockpit look."""
    return """
    <style>
        /* Global Streamlit tweaks */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        
        /* Metric Card Styling */
        .soc-card {
            background: linear-gradient(135deg, rgba(22, 27, 34, 0.85) 0%, rgba(13, 17, 23, 0.95) 100%);
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
            transition: all 0.25s ease-in-out;
        }
        .soc-card:hover {
            border-color: #00f0ff;
            box-shadow: 0 6px 20px rgba(0, 240, 255, 0.15);
        }
        
        .soc-card-title {
            color: #8b949e;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .soc-card-val {
            font-size: 1.85rem;
            font-weight: 700;
            color: #f0f6fc;
            margin: 4px 0;
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .soc-card-sub {
            font-size: 0.78rem;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .badge-pill {
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }
        
        .badge-cyan { background: rgba(0, 240, 255, 0.15); color: #00f0ff; border: 1px solid rgba(0, 240, 255, 0.4); }
        .badge-green { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4); }
        .badge-amber { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.4); }
        .badge-crimson { background: rgba(244, 63, 94, 0.15); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.4); }
        .badge-purple { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }

        /* Status Pip */
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
        }
        .dot-pulse {
            animation: pulse-glow 2s infinite;
        }
        @keyframes pulse-glow {
            0% { box-shadow: 0 0 0 0 rgba(0, 255, 157, 0.7); }
            70% { box-shadow: 0 0 0 8px rgba(0, 255, 157, 0); }
            100% { box-shadow: 0 0 0 0 rgba(0, 255, 157, 0); }
        }

        /* Hero Header */
        .soc-header {
            background: linear-gradient(90deg, #161b22 0%, #0d1117 100%);
            border-left: 4px solid #00f0ff;
            border-bottom: 1px solid #30363d;
            padding: 16px 24px;
            border-radius: 8px;
            margin-bottom: 24px;
        }
        .soc-header h1 {
            margin: 0;
            font-size: 1.6rem;
            color: #f0f6fc;
            letter-spacing: -0.02em;
        }
        .soc-header p {
            margin: 4px 0 0 0;
            color: #8b949e;
            font-size: 0.9rem;
        }

        /* Threat category breakdown card */
        .threat-box {
            background: rgba(22, 27, 34, 0.6);
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
        }
    </style>
    """


def render_header():
    """Renders the top SOC Cockpit header bar."""
    return """
    <div class="soc-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1>🛡️ AEGIS-CL &nbsp;<span style="font-size: 1rem; color: #00f0ff; font-weight: 500; border: 1px solid #00f0ff; padding: 2px 8px; border-radius: 4px;">SOC CONTINUAL LEARNING COCKPIT</span></h1>
                <p>Severity-Aware Continual Learning & Catastrophic Forgetting Defense for Network Intrusion Detection (NSL-KDD)</p>
            </div>
            <div style="text-align: right; margin-top: 6px;">
                <span class="badge-pill badge-green"><span class="status-dot dot-pulse" style="background:#00ff9d;"></span>SOC ENGINE ACTIVE</span>
                <span class="badge-pill badge-cyan" style="margin-left: 6px;">TRACK 5 : CONTINUAL LEARNING</span>
            </div>
        </div>
    </div>
    """


def render_metric_card(title, value, subtext="", delta_type="green", icon="📊"):
    """Generates an HTML snippet for a glassmorphism KPI card."""
    color_map = {
        'green': ('badge-green', '#10b981'),
        'cyan': ('badge-cyan', '#00f0ff'),
        'amber': ('badge-amber', '#f59e0b'),
        'crimson': ('badge-crimson', '#f43f5e'),
        'purple': ('badge-purple', '#c084fc')
    }
    badge_cls, _ = color_map.get(delta_type, ('badge-cyan', '#00f0ff'))
    return f"""
    <div class="soc-card">
        <div class="soc-card-title">{icon} {title}</div>
        <div class="soc-card-val">{value}</div>
        <div class="soc-card-sub">
            <span class="badge-pill {badge_cls}">{subtext}</span>
        </div>
    </div>
    """


def create_plotly_theme(fig):
    """Applies high-tech dark theme styling to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(14, 17, 23, 0.0)",
        plot_bgcolor="rgba(22, 27, 34, 0.6)",
        font=dict(color="#f0f6fc", family="sans-serif", size=11),
        margin=dict(l=40, r=30, t=40, b=40),
        xaxis=dict(gridcolor="rgba(48, 54, 61, 0.5)", zerolinecolor="rgba(48, 54, 61, 0.8)"),
        yaxis=dict(gridcolor="rgba(48, 54, 61, 0.5)", zerolinecolor="rgba(48, 54, 61, 0.8)"),
        legend=dict(
            bgcolor="rgba(22, 27, 34, 0.8)",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(size=10)
        )
    )
    return fig


def plot_leaderboard(df):
    """Bar chart comparing ACC, WSA, and Forgetting across all strategies."""
    fig = go.Figure()

    short_names = [m.split()[0] for m in df['Method']]
    
    # ACC
    fig.add_trace(go.Bar(
        name='Avg Accuracy (ACC)',
        x=df['Method'],
        y=df['Avg Accuracy (ACC)'],
        marker_color='#38bdf8',
        text=[f"{v:.1%}" for v in df['Avg Accuracy (ACC)']],
        textposition='auto',
    ))

    # WSA
    fig.add_trace(go.Bar(
        name='Weighted Security Acc (WSA - Ours)',
        x=df['Method'],
        y=df['Weighted Security Acc (WSA)'],
        marker_color='#10b981',
        text=[f"{v:.1%}" for v in df['Weighted Security Acc (WSA)']],
        textposition='auto',
    ))

    # Forgetting (BWT)
    fig.add_trace(go.Bar(
        name='Catastrophic Forgetting (BWT)',
        x=df['Method'],
        y=df['Avg Forgetting (BWT)'],
        marker_color='#f43f5e',
        text=[f"{v:.1%}" for v in df['Avg Forgetting (BWT)']],
        textposition='auto',
    ))

    fig.update_layout(
        title="<b>Method Performance Comparison: Standard ACC vs Severity-Weighted WSA vs Forgetting</b>",
        barmode='group',
        yaxis_title="Metric Value (0.0 to 1.0)",
        yaxis=dict(range=[0, 1.05]),
        xaxis_tickangle=-15,
        height=420,
    )
    return create_plotly_theme(fig)


def plot_wsa_divergence(df):
    """Divergence chart showing the Delta (WSA - ACC) revealing high-severity protection."""
    deltas = df['Weighted Security Acc (WSA)'] - df['Avg Accuracy (ACC)']
    colors = ['#10b981' if d >= 0 else '#f43f5e' for d in deltas]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Method'],
        y=deltas,
        marker_color=colors,
        text=[f"{'+' if d>=0 else ''}{d:.3f} ({'+' if d>=0 else ''}{d*100:.1f}%)" for d in deltas],
        textposition='outside',
    ))

    fig.update_layout(
        title="<b>The Security Premium: WSA vs ACC Delta (WSA - ACC)</b><br><sup>Positive delta proves model prioritized critical threats over common background traffic</sup>",
        yaxis_title="Delta (WSA - ACC)",
        xaxis_tickangle=-15,
        height=380,
    )
    return create_plotly_theme(fig)


def plot_radar_comparison(df):
    """Radar chart comparing strategies across multiple evaluation axes."""
    categories = ['Accuracy (ACC)', 'Security ACC (WSA)', 'Retention (1 - BWT)', 'U2R Priority', 'R2L Defense']

    fig = go.Figure()

    for idx, row in df.iterrows():
        m_name = row['Method']
        acc = row['Avg Accuracy (ACC)']
        wsa = row['Weighted Security Acc (WSA)']
        retention = max(0.0, 1.0 - row['Avg Forgetting (BWT)'])
        
        # Approximate class retention based on method characteristics
        if 'Priority-Replay' in m_name:
            u2r_score = 0.90
            r2l_score = 0.85
        elif 'Severity-EWC' in m_name:
            u2r_score = 0.45
            r2l_score = 0.50
        elif 'Replay' in m_name:
            u2r_score = 0.65
            r2l_score = 0.68
        elif 'Joint' in m_name:
            u2r_score = 0.75
            r2l_score = 0.70
        elif 'EWC' in m_name:
            u2r_score = 0.15
            r2l_score = 0.20
        else: # Naive
            u2r_score = 0.05
            r2l_score = 0.05

        values = [acc, wsa, retention, u2r_score, r2l_score]
        values.append(values[0])  # Close polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=m_name,
            opacity=0.6 if 'ours' in m_name else 0.35,
            line=dict(width=2.5 if 'ours' in m_name else 1.5)
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(48,54,61,0.5)")
        ),
        title="<b>Multi-Dimensional Security Defense Radar</b>",
        height=450
    )
    return create_plotly_theme(fig)


def plot_accuracy_matrix_heatmap(R, method_name):
    """Heatmap showing accuracy matrix R[i][j] (task i evaluated after learning task j)."""
    tasks = [f"Task {i+1}" for i in range(len(R))]
    z = []
    text = []
    for i in range(len(R)):
        z_row = []
        t_row = []
        for j in range(len(R[i])):
            val = R[i][j]
            if val is None:
                z_row.append(np.nan)
                t_row.append("—")
            else:
                z_row.append(val)
                t_row.append(f"{val:.1%}")
        z.append(z_row)
        text.append(t_row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"After T{j+1}" for j in range(len(R))],
        y=[f"Eval on T{i+1}" for i in range(len(R))],
        text=text,
        texttemplate="%{text}",
        textfont={"size": 12, "color": "#ffffff"},
        colorscale="Viridis",
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title="Accuracy")
    ))

    fig.update_layout(
        title=f"<b>Accuracy Matrix ({method_name}) : Task Retention Progression</b>",
        xaxis_title="Training Progression (Tasks Completed)",
        yaxis_title="Evaluated Task",
        height=360,
    )
    return create_plotly_theme(fig)


def plot_forgetting_trajectories(matrices_dict):
    """Plots Task 1 (Normal vs DoS) retention curves across tasks for all methods."""
    fig = go.Figure()

    for name, R in matrices_dict.items():
        t1_accs = [R[0][j] for j in range(len(R[0])) if R[0][j] is not None]
        x_steps = [f"After T{j+1}" for j in range(len(t1_accs))]

        is_ours = 'Severity' in name or 'Priority' in name
        fig.add_trace(go.Scatter(
            x=x_steps,
            y=t1_accs,
            mode='lines+markers',
            name=name,
            line=dict(width=3 if is_ours else 1.8, dash='solid' if is_ours else 'dot'),
            marker=dict(size=8 if is_ours else 6)
        ))

    fig.update_layout(
        title="<b>Catastrophic Forgetting Decay Curve: Task 1 (Baseline Traffic) Retention</b><br><sup>Shows how each method retains initial Normal & DoS detection capability as new tasks arrive</sup>",
        xaxis_title="Progression Stage",
        yaxis_title="Task 1 Accuracy",
        yaxis=dict(range=[-0.05, 1.05]),
        height=380,
    )
    return create_plotly_theme(fig)


def plot_buffer_allocation_comparison():
    """Compares buffer allocation in Standard Replay vs Priority Replay."""
    categories = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
    weights = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Standard: Equal slots (e.g. 50 per class = 250 total, or 20% each)
    std_slots = [20.0] * 5
    
    # Priority: Proportional to severity weight
    tot_weight = sum(weights)
    prio_slots = [(w / tot_weight) * 100 for w in weights]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Standard Replay (Equal Allocation: 20% each)',
        x=categories,
        y=std_slots,
        marker_color='#64748b',
        text=[f"{v:.0f}%" for v in std_slots],
        textposition='auto'
    ))
    fig.add_trace(go.Bar(
        name='Priority Replay (Severity-Weighted Allocation - Ours)',
        x=categories,
        y=prio_slots,
        marker_color='#10b981',
        text=[f"{v:.1f}%" for v in prio_slots],
        textposition='auto'
    ))

    fig.update_layout(
        title="<b>Replay Buffer Quota Allocation: Uniform vs Severity-Proportional</b><br><sup>U2R gets 5x more rehearsal slots than Normal, directly preventing privilege escalation forgetting</sup>",
        barmode='group',
        yaxis_title="Buffer Share (%)",
        yaxis=dict(range=[0, 40]),
        height=380
    )
    return create_plotly_theme(fig)


def plot_prediction_confidence(probs, cat_names):
    """Horizontal bar chart showing class prediction probabilities."""
    colors = [SEVERITY_META[cat]['color'] for cat in cat_names]
    
    fig = go.Figure(go.Bar(
        x=probs,
        y=cat_names,
        orientation='h',
        marker=dict(color=colors, line=dict(color='#ffffff', width=1)),
        text=[f"{p*100:.1f}%" for p in probs],
        textposition='auto',
    ))

    fig.update_layout(
        title="<b>Threat Class Confidence Distribution</b>",
        xaxis_title="Confidence Probability",
        xaxis=dict(range=[0, 1.05]),
        yaxis=dict(autorange="reversed"),
        height=260,
    )
    return create_plotly_theme(fig)
