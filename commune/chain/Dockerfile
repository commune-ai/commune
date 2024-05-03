from ghcr.io/agicommies/subspace:dd5a1e5

# install wget 

RUN apt-get update && apt-get install -y wget


RUN wget https://github.com/agicommies/communex/raw/main/docs/bootnodes.txt -P /node-data
RUN wget https://github.com/agicommies/subspace/raw/current/specs/main.json -P /node-data/specs

# store the bootnodes in a in the local path of where this script is locaed
# this is to ensure that the bootnodes are not lost when the container is restarted
# and the bootnodes are not lost

# NODE_DATA_DIR="${NODE_DATA_DIR:-/node-data}"
# BOOTNODES="$(< "$NODE_DATA_DIR/bootnodes.txt" xargs echo)"