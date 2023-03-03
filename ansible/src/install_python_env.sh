#!/bin/bash

# This script installs the python environment for the commun project.

if [[ "$1" == "venv" ]]; then
    echo "Entering Virtual Environment (communenv)"
    if [ -d "venv" ]; then
        echo "communenv already exists, skipping, creating new one"
    else
        echo "communenv does not exist, creating new one"
        python3 -m venv venv
    fi

    source communenv/bin/activate

else
    echo "Skipping Virtual Environment"
fi


# Upgrade pip

python3 -m pip install --upgrade pip

# Install dependencies

python3 -m pip install bittensor
python3 -m git clone https://github.com/opentensor/cubit.git /cubit
python3 -m pip install -e /cubit
python3 -m pip install --upgrade torch
python3 -m pip install -e .
python3 -m pip install pytest
python3 -m pip install jupyterlab
python3 -m pip install accelerate

