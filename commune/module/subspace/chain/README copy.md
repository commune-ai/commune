# Running a Commune Node

## Requirements

- [Docker](https://docs.docker.com/get-docker/)
- Docker Compose
- `wget`
- [`yq`](https://mikefarah.gitbook.io/yq/)

Installing on Ubuntu:

```sh
sudo apt install docker.io docker-compose
sudo apt install wget

# Ubuntu has `yq` on apt since 23.04
sudo apt install yq
# On older versions, you can install it with:
sudo wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && \
    sudo chmod +x /usr/bin/yq
```

## Docker Image

We provide the laster Docker image for the Commune node in [this Github
package][docker-package] at the [`latest` tag][docker-image]. You can pull the
image with the command:

```sh
docker pull ghcr.io/agicommies/subspace:dd5a1e5
```

## Chain specs

You can get the chain specs file from the [`main/specs/main.json` file] at
[github.com/agicommies/subspace/blob/current/specs/main.json](https://github.com/agicommies/subspace/blob/current/specs/main.json):

```sh
wget https://github.com/agicommies/subspace/raw/current/specs/main.json
```

## Getting bootnodes

You can currently get the bootnodes list with:

```sh
wget https://github.com/agicommies/communex/raw/main/docs/bootnodes.txt
```

## Deploying the config

```sh
# cd into this directory if you haven't already
cd docs/running_a_node/
```

Pick a directory to store the node data, and copy the chain specs, bootnodes
list and start script into it:

```sh
export COMMUNE_NODE_DIR="/commune-node"
echo COMMUNE_NODE_DIR=$COMMUNE_NODE_DIR > .env

mkdir -p "$COMMUNE_NODE_DIR/specs"

cp main.json "$COMMUNE_NODE_DIR/specs/main.json"
cp bootnodes.txt "$COMMUNE_NODE_DIR/bootnodes.txt"
cp node-start.sh "$COMMUNE_NODE_DIR/node-start.sh"
```

### Note

- Make sure that the `COMMUNE_NODE_DIR` variable is correctly set in your shell
  and in the `.env` file (used by `docker-compose`).
- Make sure you don't have old node data in that directory if you want to run
  the node fresh!

## Running the node

The provided [`./docker-compose.yml` file] expects the environment variable
`COMMUNE_NODE_DIR` to be set to the node data directory, that will be mounted to
`/node-data` inside the container. It will serve the P2P connection at `30333`
and the RPC service at `9933`.

```sh
docker-compose up -d
```

By default the node will work in archive mode, which means it will store all the
chain data (gigabytes). This can be changed by replacing the `--pruning=archive`
and `--sync=full` flags with `--sync=warp` in the `docker-compose.yml` script.

Now you can track the telemetry using [this link](http://telemetry.communeai.net/#/0xbc6eb9753e2417476601485f9f8ef8474701ec199d456f989bd397682c9425c5)

---

References:

- [docs.substrate.io/reference/command-line-tools/node-template/](https://docs.substrate.io/reference/command-line-tools/node-template/)

[docker-package]: https://github.com/orgs/agicommies/packages/container/package/subspace
[docker-image]: https://github.com/orgs/agicommies/packages/container/subspace/164109015?tag=latest

[`commune/modules/subspace/chain/chain.yaml` file]: https://github.com/commune-ai/commune/blob/main/commune/modules/subspace/chain/chain.yaml
[`main/specs/main.json` file]: https://github.com/commune-ai/subspace/blob/main/specs/main.json

[`./docker-compose.yml` file]: ./docker-compose.yml
