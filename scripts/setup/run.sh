#!/usr/bin/env bash
set -euo pipefail # Exit on error, unset variables, pipefail

# Tìm thu m?c g?c c?a d? án
DIR=$(cd "$(dirname "$0")/.." && pwd)

# Di chuy?n vào thu m?c g?c c?a d? án
cd "$DIR"

# Kích ho?t virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
else
    echo "Error: Virtual environment '.venv' not found. Please run 'scripts/setup_env.sh' first."
    exit 1
fi

# Ch?y ?ng d?ng chính
echo "Starting VoIP AI Agent application..."
python3 -m app.main