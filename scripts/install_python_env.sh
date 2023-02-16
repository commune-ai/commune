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

pip install --upgrade pip

# Install dependencies

pip install bittensor
git clone https://github.com/opentensor/cubit.git /cubit
pip install -e /cubit
pip install --upgrade torch
pip install -e .
pip install pytest
pip install jupyterlab
pip install accelerate

