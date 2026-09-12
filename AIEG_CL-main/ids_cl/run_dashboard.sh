#!/usr/bin/env bash
# AEGIS-CL: Dashboard Launcher Script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_EXEC="/home/prajwal/Traffic_Baseine/traffic_baseline/dl/bin/python3"
STREAMLIT_EXEC="/home/prajwal/Traffic_Baseine/traffic_baseline/dl/bin/streamlit"

echo "================================================================="
echo "🛡️  AEGIS-CL: Severity-Aware Continual Learning SOC Cockpit"
echo "================================================================="
echo "Python:    $PYTHON_EXEC"
echo "Streamlit: $STREAMLIT_EXEC"
echo "Directory: $SCRIPT_DIR"
echo "Port:      8501"
echo "================================================================="

$STREAMLIT_EXEC run app.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.base "dark"
