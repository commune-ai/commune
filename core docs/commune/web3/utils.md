# Project - Custom Web3 Providers

**Overview:**
This library provides custom HTTP and WebSocket providers for the [Web3.py](https://github.com/ethereum/web3.py) Ethereum library. This library contains additional features to control the `requests` session parameters such as the connection pool, and handles the session cache.

## Requirements
- Python 3.6 and above
- Web3.py

## Usage

This library provides the following classes and functions:

**Classes:**

- `CustomHTTPProvider`: This class extends the `HTTPProvider` class from the web3 library. It adds features to control the `requests` session parameters and overrides the `make_request` method.

**Functions:**

- `get_web3`: This function returns a web3 instance connected via the given network_url. It adds POA (Proof of Authority) middleware when connecting to the Rinkeby Testnet.

- `get_web3_connection_provider`: This function returns the suitable web3 provider based on the network_url.

- `make_post_request`: This function makes a POST request to the given URI and data, handling connection pool and session cache.

**Running the code:**

This module can not be run directly after cloning the repository as it is a library module. The functions provided in this library can be used in your Ethereum-centric Python projects.

## Disclaimer:
This library is built for education purposes, do not use it for illegal activities or production systems without proper testing. Please use it responsibly and ethically.

## Copyright 
This project is developed by Ocean Protocol Foundation and is released under the Apache 2.0 License.
The code is open-source and found at [Ocean Protocol Github Repository](https://github.com/oceanprotocol/).