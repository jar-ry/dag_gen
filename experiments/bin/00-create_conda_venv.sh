#!/bin/bash
set -e
echo "==== Creating conda environment ====="
echo "Creating conda environment from 'conda-linux.yaml'"
# find linux username
CONDA_ENV="py37_env"

# create Conda environment with name specific to user
conda env create -f conda_linux.yaml --name=$CONDA_ENV

echo "Environment: ${CONDA_ENV} created"


