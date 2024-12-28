# Blockchain Integration

Commune provides robust blockchain integration capabilities, with primary support for Subtensor and extensibility for other chains. This guide covers the integration patterns and features available for blockchain interaction.

## Subtensor Integration

### 1. Basic Setup

```python
import commune as c
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class NetworkConfig:
    """Configuration for Subtensor network."""
    network: str
    netuid: int
    stake: float
    endpoint: Optional[str] = None

class SubtensorModule(c.Module):
    """Module for interacting with Subtensor network."""
    
    def __init__(
        self,
        config: Optional[NetworkConfig] = None
    ):
        super().__init__()
        self.config = config or NetworkConfig(
            network='mainnet',
            netuid=1,
            stake=0.1
        )
        self.subtensor = c.module('subtensor')(
            network=self.config.network,
            endpoint=self.config.endpoint
        )
    
    async def get_network_info(self) -> Dict[str, Any]:
        """Get current network information."""
        try:
            return {
                'block': await self.subtensor.get_current_block(),
                'stake': await self.subtensor.get_stake(),
                'peers': await self.subtensor.get_peers(),
                'difficulty': await self.subtensor.get_difficulty()
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def get_network_status(self) -> Dict[str, Any]:
        """Get network health status."""
        try:
            return {
                'is_synced': await self.subtensor.is_synced(),
                'connections': await self.subtensor.get_connected_peers(),
                'version': await self.subtensor.get_version()
            }
        except Exception as e:
            return {'error': str(e)}

# Example usage
async def main():
    module = SubtensorModule()
    info = await module.get_network_info()
    print(f"Network Info: {info}")
```

## Validator System

### 1. Basic Validator

```python
import commune as c
from typing import Dict, Any, List
from dataclasses import dataclass
import asyncio

@dataclass
class ValidatorConfig:
    """Configuration for validator module."""
    network: str
    stake: float
    interval: int
    min_score: float

class CustomValidator(c.Module):
    """Custom validator implementation."""
    
    def __init__(
        self,
        config: Optional[ValidatorConfig] = None
    ):
        super().__init__()
        self.config = config or ValidatorConfig(
            network='local',
            stake=1.0,
            interval=12,
            min_score=0.5
        )
        self.validator = c.module('validator')(
            network=self.config.network,
            stake=self.config.stake
        )
    
    async def score_module(
        self,
        module: c.Module
    ) -> float:
        """Score a module based on performance metrics."""
        try:
            # Get module metrics
            latency = await self.measure_latency(module)
            uptime = await self.get_uptime(module)
            quality = await self.assess_quality(module)
            
            # Calculate weighted score
            score = (
                0.3 * self.normalize(latency, max_val=1000) +
                0.3 * uptime +
                0.4 * quality
            )
            
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.0
    
    async def measure_latency(
        self,
        module: c.Module
    ) -> float:
        """Measure module response latency."""
        start = asyncio.get_event_loop().time()
        try:
            await module.ping()
            return asyncio.get_event_loop().time() - start
        except Exception:
            return float('inf')
    
    async def get_uptime(
        self,
        module: c.Module
    ) -> float:
        """Get module uptime score."""
        try:
            stats = await module.get_stats()
            return stats.get('uptime', 0) / 100.0
        except Exception:
            return 0.0
    
    async def assess_quality(
        self,
        module: c.Module
    ) -> float:
        """Assess module output quality."""
        try:
            # Run test queries
            results = await self.run_test_suite(module)
            return sum(results) / len(results)
        except Exception:
            return 0.0
    
    @staticmethod
    def normalize(
        value: float,
        max_val: float
    ) -> float:
        """Normalize value to 0-1 range."""
        return max(0.0, min(1.0, 1 - (value / max_val)))
    
    async def run_test_suite(
        self,
        module: c.Module
    ) -> List[float]:
        """Run test cases and return scores."""
        test_cases = [
            ('ping', {}),
            ('get_info', {}),
            ('process', {'data': 'test'})
        ]
        
        results = []
        for method, args in test_cases:
            try:
                await getattr(module, method)(**args)
                results.append(1.0)
            except Exception:
                results.append(0.0)
        
        return results

# Example usage
async def main():
    validator = CustomValidator()
    module = await c.connect('test_module')
    score = await validator.score_module(module)
    print(f"Module Score: {score}")
```

## Best Practices

### 1. Error Handling

```python
import commune as c
from typing import Dict, Any, Optional
import asyncio

class BlockchainModule(c.Module):
    """Base class for blockchain interactions."""
    
    async def safe_call(
        self,
        method: str,
        *args,
        retries: int = 3,
        delay: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Make safe blockchain calls with retries."""
        for attempt in range(retries):
            try:
                result = await getattr(self, method)(
                    *args,
                    **kwargs
                )
                return {
                    'success': True,
                    'result': result,
                    'attempts': attempt + 1
                }
            except Exception as e:
                if attempt == retries - 1:
                    return {
                        'success': False,
                        'error': str(e),
                        'attempts': attempt + 1
                    }
                await asyncio.sleep(delay)
                continue
    
    @staticmethod
    def validate_transaction(
        tx: Dict[str, Any]
    ) -> Optional[str]:
        """Validate transaction parameters."""
        required = ['to', 'value', 'data']
        
        # Check required fields
        missing = [f for f in required if f not in tx]
        if missing:
            return f"Missing fields: {', '.join(missing)}"
        
        # Validate value
        try:
            value = float(tx['value'])
            if value <= 0:
                return "Value must be positive"
        except ValueError:
            return "Invalid value format"
        
        return None

# Example usage
async def main():
    module = BlockchainModule()
    result = await module.safe_call(
        'send_transaction',
        to='0x...',
        value=1.0,
        data='0x...'
    )
    print(result)
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
