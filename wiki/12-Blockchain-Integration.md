# Blockchain Integration

Commune provides robust blockchain integration capabilities, with primary support for Subtensor and extensibility for other chains. This guide covers the integration patterns and features available for blockchain interaction.

## Subtensor Integration

### 1. Basic Setup

```python
import commune as c

# Initialize Subtensor connection
subtensor = c.module('subtensor')

# Connect to specific network
subtensor = c.module('subtensor', network='mainnet')
```

### 2. Network Operations

```python
# Get network info
info = subtensor.get_network_info()

# Check network status
status = subtensor.get_network_status()

# Get current block
block = subtensor.get_current_block()
```

## Validator System

### 1. Basic Validator

```python
class MyValidator(c.Module):
    def __init__(self, network='local'):
        super().__init__()
        self.vali = c.module('vali')(
            network=network,
            score=self.score_fn
        )
    
    def score_fn(self, module):
        # Implement validation logic
        return score  # 0.0 to 1.0
```

### 2. Advanced Validation

```python
class CustomValidator(c.Vali):
    def __init__(self):
        super().__init__(
            network='subspace',
            batch_size=128,
            timeout=3
        )
    
    async def score(self, module):
        try:
            # Custom validation logic
            result = await module.validate()
            return self.normalize_score(result)
        except Exception as e:
            return 0.0
```

## Token Economics

### 1. Staking Operations

```python
# Stake tokens
result = subtensor.stake(
    amount=1000,
    target_address='address'
)

# Unstake tokens
result = subtensor.unstake(
    amount=500,
    source_address='address'
)
```

### 2. Rewards System

```python
class RewardModule(c.Module):
    def calculate_rewards(self, scores):
        # Implement reward distribution logic
        return {
            'address1': score1 * weight1,
            'address2': score2 * weight2
        }
    
    def distribute_rewards(self, rewards):
        for address, amount in rewards.items():
            subtensor.transfer(address, amount)
```

## Network Types

### 1. Local Network
```python
# Start local validator
validator = c.module('vali')(
    network='local',
    subnet='test'
)
```

### 2. Testnet
```python
# Connect to testnet
validator = c.module('vali')(
    network='subspace:test',
    subnet='main'
)
```

### 3. Mainnet
```python
# Connect to mainnet
validator = c.module('vali')(
    network='subspace:main',
    subnet='main'
)
```

## Consensus Mechanisms

### 1. Voting System

```python
class VotingModule(c.Module):
    def __init__(self):
        self.voting_networks = ['bittensor', 'subspace']
        self.epoch_time = 0
        self.vote_time = 0
    
    async def vote(self, candidates):
        scores = await self.evaluate_candidates(candidates)
        return self.submit_votes(scores)
```

### 2. Weight Management

```python
# Set weights for network
subtensor.set_weights(
    weights=weights_matrix,
    uids=uid_list,
    netuid=0
)

# Get current weights
current_weights = subtensor.get_weights()
```

## Security Features

### 1. Key Management

```python
# Generate new key
key = c.get_key(create=True)

# Set validator key
validator.set_key(key)

# Secure operations
signed_tx = key.sign_transaction(tx_data)
```

### 2. Access Control

```python
class SecureValidator(c.Vali):
    def __init__(self):
        super().__init__()
        self.require_stake = True
        self.min_stake = 1000
    
    def verify_access(self, address):
        stake = self.get_stake(address)
        return stake >= self.min_stake
```

## Performance Optimization

### 1. Batch Processing

```python
class BatchValidator(c.Vali):
    def __init__(self):
        super().__init__(batch_size=128)
    
    async def process_batch(self, modules):
        futures = [self.score(m) for m in modules]
        return await c.gather(futures)
```

### 2. Caching

```python
class CachedValidator(c.Vali):
    def __init__(self):
        super().__init__()
        self.cache = {}
        self.cache_timeout = 3600
    
    async def get_score(self, module):
        if module in self.cache:
            return self.cache[module]
        score = await self.calculate_score(module)
        self.cache[module] = score
        return score
```

## Monitoring and Analytics

### 1. Network Metrics

```python
# Get network statistics
stats = subtensor.get_network_stats()

# Monitor validator performance
performance = validator.get_performance_metrics()
```

### 2. Event Tracking

```python
class MetricsValidator(c.Vali):
    async def track_events(self):
        async for event in self.subscribe_events():
            self.process_event(event)
    
    def process_event(self, event):
        # Log and analyze event
        c.log.info(f"Event: {event}")
```

## Best Practices

1. **Validation Design**
   - Implement fair and transparent scoring
   - Handle edge cases gracefully
   - Include proper error handling

2. **Security**
   - Secure key management
   - Implement proper access controls
   - Regular security audits

3. **Performance**
   - Use batch processing when possible
   - Implement caching strategies
   - Monitor resource usage

## Common Issues

1. **Network Connectivity**
```python
# Check network connection
status = subtensor.check_connection()

# Reconnect if needed
if not status:
    subtensor.reconnect()
```

2. **Validation Errors**
```python
# Handle validation failures
try:
    score = await validator.score(module)
except Exception as e:
    c.log.error(f"Validation error: {e}")
    score = 0.0
```

## Next Steps

1. Explore [Custom Validators](40-Custom-Validators.md)
2. Learn about [Token Economics](41-Token-Economics.md)
3. Study [Security Best Practices](43-Security-Best-Practices.md)
