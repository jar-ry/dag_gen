#!/usr/bin/env bash - l

# !!! You have to run "source ./bin/01-activate_conda_env" in terminal !!!
CONDA_ENV="py37_env"

# activate arm env
conda activate $CONDA_ENV

echo "Environment: ${CONDA_ENV} activated"

# databricks-connect requires these to be unset
unset PYSPARK_PYTHON
unset SPARK_HOME

# need longer time to work than default 10min
export TMOUT=43200