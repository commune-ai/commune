#!/bin/bash
export NODE_OPTIONS=--openssl-legacy-provider
ganache-cli --port ${GANACHE_PORT} -h 0.0.0.0 --gasLimit 12000000 --accounts 10 --hardfork istanbul --mnemonic brownie --fork https://mainnet.infura.io/v3/${WEB3_INFURA_PROJECT_ID} --chainId 31337

