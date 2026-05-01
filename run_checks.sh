#!/bin/bash

echo "Running ThermoPINN checks..."

python -c "from app.main import app; print('Import check passed')"

pytest

echo "All checks completed."
