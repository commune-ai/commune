
# Vali - Validator Module for Distributed Networks

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Vali is a Python module designed for validating and scoring modules in distributed networks. It provides functionality for running validation epochs, scoring modules, and maintaining a scoreboard of module performance.

## Features

- Connect to different networks (local, test, main)
- Score modules based on custom scoring functions
- Run validation epochs at specified intervals
- Maintain a scoreboard of module performance
- Sign and verify module scores with cryptographic proofs
- Vote on modules in voting-enabled networks

## Installation

```bash
pip install commune
```

## Quick Start

```python
import commune as c
from commune.vali import Vali

# Create a validator with a custom scoring function
def my_score_function(client):
    # Implement your scoring logic here
    return client.info().get('score', 0)

# Initialize the validator
validator = Vali(
    network='local',
    score=my_score_function,
    tempo=10,  # Run epochs every 10 seconds
    verbose=True
)

# Run a single epoch manually
results = validator.epoch()

# Get current stats
stats = validator.stats()
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `network` | str | 'server' | Network to connect to ('local', 'test', 'main') |
| `search` | str | None | Optional search string to filter modules |
| `batch_size` | int | 128 | Batch size for parallel tasks |
| `score` | callable/int | None | Scoring function for modules |
| `key` | str | None | Key for the module |
| `tempo` | int | 10 | Time between epochs (seconds) |
| `max_sample_age` | int | 3600 | Maximum age of samples (seconds) |
| `timeout` | int | 3 | Timeout per module evaluation (seconds) |
| `update` | bool | True | Update during the first epoch |
| `run_loop` | bool | True | Run the validation loop automatically |
| `verbose` | bool | True | Print verbose output |
| `path` | str | None | Storage path for module evaluation results |

## Core Methods

### `epoch(**kwargs)`
Runs a validation epoch, scoring all modules in the network.

```python
# Run an epoch with updated network settings
results = validator.epoch(network='test', search='model')
```

### `score_module(module, **kwargs)`
Scores an individual module and stores the result.

```python
# Score a specific module
module_info = validator.score_module(module_key)
```

### `stats(keys=['name', 'score', 'duration', 'url', 'key', 'time', 'age'], ascending=True, by='score', to_dict=False, page=None, max_age=1000, update=False, **kwargs)`
Returns statistics about the scored modules.

```python
# Get stats sorted by score in descending order
stats = validator.stats(ascending=False, by='score')
```

### `vote(results)`
Submits votes for modules in voting-enabled networks.

```python
# Vote on modules based on epoch results
validator.vote(results)
```

### `set_network(network=None, tempo=None, search=None, path=None, update=False)`
Configures the network connection.

```python
# Change network settings
validator.set_network(network='main', tempo=30)
```

### `set_score(score)`
Sets the scoring function for modules.

```python
# Update the scoring function
validator.set_score(new_score_function)
```

## Example: Custom Scoring Function

```python
from commune.vali import Vali

def advanced_score(client, **kwargs):
    # Get module info
    info = client.info()
    
    # Check if module is responsive
    if not info:
        return 0
    
    # Calculate score based on multiple factors
    base_score = 10
    uptime_bonus = info.get('uptime', 0) / 100
    performance = info.get('performance', 0) / 10
    
    return base_score + uptime_bonus + performance

# Create validator with custom scoring
validator = Vali(
    network='test',
    score=advanced_score,
    tempo=30,
    batch_size=64
)
```

## Running as a Standalone Validator

```python
from commune.vali import Vali

# Run a single epoch without starting the continuous loop
results = Vali.run_epoch(network='local')
print(results)
```

## Storage and Persistence

Vali stores module evaluation results in a configurable storage path. By default, this is in the commune storage directory under `/vali/{network}/{subnet}`.

```python
# Get the path where module data is stored
storage_path = validator.path
```

## License

MIT License
