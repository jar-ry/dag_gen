#!/bin/bash
set -e
echo "==== Installing pip packages (dev tools) ====="
echo "command: pip install -r requirements-dev.txt --force-reinstall"
pip install -r requirements.txt --force-reinstall
pip install .