# Polaris Validator System

The Polaris Validator System is a modular implementation for validating miners on various compute networks, including Bittensor and Commune. The system verifies miner registration, validates compute resources through SSH connections, tracks container usage, and submits weights to the respective networks.

## Features

- **Network-agnostic design**: Support for multiple networks through a common interface
- **SSH-based resource validation**: Directly verifies compute resources on miner machines
- **Container usage tracking**: Tracks active containers and their resource utilization
- **Configurable scoring algorithm**: Customizable parameters for hardware and container usage scoring
- **Modular architecture**: Easy to extend with new network implementations

## Architecture

The system is organized into the following components:

- **Configuration**: Centralized configuration with environment variable and command-line support
- **Validators**: Network-specific validators that implement a common interface
- **Utilities**: Shared functionality for API clients, SSH connections, etc.
- **Resource Scoring**: Algorithms for scoring hardware resources and container usage

## Directory Structure

```
validator/
├── src/
│   ├── config.py              # Configuration management
│   ├── main.py                # Main entry point
│   ├── simplified_validator.py # Simplified standalone validator for testing
│   ├── utils/                 # Utility modules
│   │   ├── api_client.py      # API client for Polaris API
│   │   ├── firebase_client.py # Firebase client for data access
│   │   ├── logging_utils.py   # Logging utilities
│   │   ├── resource_scoring.py # Resource scoring algorithms
│   │   └── ssh_utils.py       # SSH utilities for miner connections
│   └── validators/            # Validator implementations
│       ├── base_validator.py  # Base validator interface
│       ├── bittensor_validator.py # Bittensor-specific validator
│       └── validator_factory.py # Factory for creating validators
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r validator/requirements.txt
   ```

2. Set up environment variables or use command-line arguments to configure the validator.

## Running the Validator

### Full System

To run the full validator system:

```bash
python -m validator.src.main --wallet_name YOUR_WALLET --hotkey YOUR_HOTKEY --netuid 33 --network test
```

### Simplified Validator (for testing)

For quick testing, you can use the simplified validator:

```bash
python -m validator.src.simplified_validator --wallet_name YOUR_WALLET --hotkey YOUR_HOTKEY --netuid 33 --network test
```

## Configuration Options

The validator system can be configured through environment variables or command-line arguments:

### Bittensor Settings
- `--wallet_name`: Bittensor wallet name
- `--hotkey`: Bittensor hotkey name
- `--netuid`: Subnet UID (default: 33)
- `--network`: Network name ('mainnet', 'testnet', 'finney', 'test')

### Validation Settings
- `--validation_interval`: Seconds between validations (default: 3600)
- `--submission_interval`: Seconds between weight submissions (default: 3600)
- `--log_level`: Logging level (default: INFO)
- `--log_file`: Log file path (default: validator.log)
- `--max_weight_value`: Maximum weight value (default: 1.0)
- `--min_score_for_weight`: Minimum score required to receive weight (default: 5.0)

## Extending the System

To add support for a new network:

1. Create a new validator class that extends `BaseValidator`
2. Implement the required abstract methods
3. Add the new validator to the `ValidatorFactory`
