#/bin/bash

# INSTALL PACKAGES BEFORE COMMUNE (This is a hack to get around the bittensor install)
pip install bittensor
# BITTENSOR FIXES FOR NOW
pip install --upgrade substrate-interface
pip install --upgrade torch

# INSTALL COMMUNE
COPY ./commune /app/commune
COPY ./scripts /app/scripts
COPY ./requirements.txt /app/requirements.txt
COPY ./setup.py /app/setup.py
COPY ./README.md /app/README.md

# INSTALL PACKAGES AFTER COMMUNE
pip install -e .
pip install openai
pip install google-search-results
pip install wikipedia
pip install pytest
pip install jupyterlab
pip install accelerate



