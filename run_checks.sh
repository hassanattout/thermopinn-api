#!/bin/bash
set -euo pipefail

echo "Running ThermoPINN checks..."

python -c "from app.main import app; print('Import check passed')"

pytest -q

echo "All checks completed."
