# Tutorial: Implementing Custom Validators

This tutorial guides you through creating custom validators in Commune, essential for network consensus and quality assurance.

## Prerequisites
- Basic understanding of Commune modules
- Familiarity with async/await in Python
- Knowledge of basic blockchain concepts

## Understanding Validators

Validators in Commune serve several purposes:
1. Quality assessment of network modules
2. Network consensus participation
3. Reward distribution
4. Network security maintenance

## Basic Validator Structure

Let's create a simple validator for API modules:

```python
import commune as c
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass
import asyncio
import time

@dataclass
class ValidatorConfig:
    """Configuration for API validator."""
    network: str = 'local'
    subnet: Optional[str] = None
    batch_size: int = 32
    timeout: int = 5
    min_response_time: float = 0.1
    max_response_time: float = 2.0
    required_endpoints: List[str] = None
    
    def __post_init__(self):
        """Set default required endpoints if none provided."""
        if self.required_endpoints is None:
            self.required_endpoints = ['info', 'status', 'health']

class APIValidator(c.Vali):
    """Validator for API modules with comprehensive scoring."""
    
    def __init__(
        self,
        config: Optional[ValidatorConfig] = None
    ):
        """Initialize validator with configuration.
        
        Args:
            config: Validator configuration
        """
        self.config = config or ValidatorConfig()
        
        super().__init__(
            network=self.config.network,
            subnet=self.config.subnet,
            batch_size=self.config.batch_size,
            timeout=self.config.timeout,
            score=self.score_api,
            run_loop=True
        )
        
        # Initialize metrics storage
        self.metrics: Dict[str, Dict[str, Any]] = {}
    
    async def score_api(
        self,
        module: c.Module
    ) -> float:
        """Score an API module based on multiple criteria.
        
        Args:
            module: Module to score
            
        Returns:
            Score between 0 and 1
        """
        try:
            # Initialize scoring components
            endpoint_score = await self._check_endpoints(module)
            response_score = await self._measure_response(module)
            error_score = await self._test_error_handling(module)
            
            # Calculate weighted final score
            final_score = (
                0.4 * endpoint_score +
                0.4 * response_score +
                0.2 * error_score
            )
            
            # Update metrics
            self._update_metrics(module, {
                'endpoint_score': endpoint_score,
                'response_score': response_score,
                'error_score': error_score,
                'final_score': final_score
            })
            
            return final_score
        except Exception as e:
            print(f"Error scoring module {module}: {e}")
            return 0.0
    
    async def _check_endpoints(
        self,
        module: c.Module
    ) -> float:
        """Check required endpoint availability.
        
        Args:
            module: Module to check
            
        Returns:
            Score based on endpoint availability
        """
        available = 0
        total = len(self.config.required_endpoints)
        
        for endpoint in self.config.required_endpoints:
            try:
                if hasattr(module, endpoint):
                    # Test endpoint
                    await getattr(module, endpoint)()
                    available += 1
            except Exception:
                continue
        
        return available / total if total > 0 else 0.0
    
    async def _measure_response(
        self,
        module: c.Module
    ) -> float:
        """Measure module response times.
        
        Args:
            module: Module to measure
            
        Returns:
            Score based on response times
        """
        times = []
        
        for _ in range(3):  # Make multiple measurements
            try:
                start = time.time()
                await module.info()  # Use info endpoint as benchmark
                elapsed = time.time() - start
                times.append(elapsed)
            except Exception:
                times.append(self.config.max_response_time)
        
        # Calculate average response time
        avg_time = sum(times) / len(times)
        
        # Normalize to score between 0 and 1
        if avg_time <= self.config.min_response_time:
            return 1.0
        elif avg_time >= self.config.max_response_time:
            return 0.0
        else:
            return 1.0 - (
                (avg_time - self.config.min_response_time) /
                (self.config.max_response_time - self.config.min_response_time)
            )
    
    async def _test_error_handling(
        self,
        module: c.Module
    ) -> float:
        """Test module's error handling capabilities.
        
        Args:
            module: Module to test
            
        Returns:
            Score based on error handling
        """
        tests = [
            self._test_invalid_input(module),
            self._test_timeout_handling(module),
            self._test_error_response(module)
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        return sum(1.0 for r in results if r is True) / len(tests)
    
    async def _test_invalid_input(
        self,
        module: c.Module
    ) -> bool:
        """Test handling of invalid input.
        
        Args:
            module: Module to test
            
        Returns:
            True if handles invalid input correctly
        """
        try:
            await module.process(None)
            return False  # Should have raised an error
        except Exception:
            return True
    
    async def _test_timeout_handling(
        self,
        module: c.Module
    ) -> bool:
        """Test handling of timeouts.
        
        Args:
            module: Module to test
            
        Returns:
            True if handles timeouts correctly
        """
        try:
            async with asyncio.timeout(0.1):
                await module.process("timeout_test")
            return True
        except asyncio.TimeoutError:
            return True  # Timeout is expected
        except Exception:
            return False
    
    async def _test_error_response(
        self,
        module: c.Module
    ) -> bool:
        """Test error response format.
        
        Args:
            module: Module to test
            
        Returns:
            True if error response is properly formatted
        """
        try:
            result = await module.process("error_test")
            if isinstance(result, dict) and 'error' in result:
                return True
            return False
        except Exception as e:
            return hasattr(e, 'error_code')
    
    def _update_metrics(
        self,
        module: c.Module,
        scores: Dict[str, float]
    ) -> None:
        """Update module metrics history.
        
        Args:
            module: Scored module
            scores: Dictionary of scores
        """
        module_id = str(module)
        
        try:
            if module_id not in self.metrics:
                self.metrics[module_id] = {
                    'history': [],
                    'average_score': 0.0
                }
            
            # Add new scores
            self.metrics[module_id]['history'].append(scores)
            
            # Keep last 100 scores
            if len(self.metrics[module_id]['history']) > 100:
                self.metrics[module_id]['history'].pop(0)
            
            # Update average
            self.metrics[module_id]['average_score'] = (
                sum(h['final_score'] for h in self.metrics[module_id]['history']) /
                len(self.metrics[module_id]['history'])
            )
        except Exception as e:
            print(f"Error updating metrics for {module_id}: {e}")

# Example usage
async def main():
    """Run validator example."""
    try:
        # Create validator with custom config
        config = ValidatorConfig(
            batch_size=16,
            timeout=3,
            min_response_time=0.05,
            max_response_time=1.0
        )
        validator = APIValidator(config)
        
        # Connect to test module
        module = await c.connect('test_api')
        
        # Score module
        score = await validator.score_api(module)
        print(f"Module score: {score}")
        
        # Get metrics
        metrics = validator.metrics[str(module)]
        print(f"Module metrics: {metrics}")
    except Exception as e:
        print(f"Error in validation: {e}")

if __name__ == '__main__':
    asyncio.run(main())

## Advanced Validator Features

### 1. Custom Scoring Logic

```python
class MLValidator(c.Vali):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_test_data()
    
    def load_test_data(self):
        """Load test dataset for validation."""
        self.test_data = c.get('validator/test_data', default=[])
    
    async def score(self, module) -> float:
        scores = []
        
        for test_case in self.test_data:
            try:
                # Test prediction quality
                result = await module.predict(test_case['input'])
                accuracy = self.calculate_accuracy(
                    result, 
                    test_case['expected']
                )
                scores.append(accuracy)
            except Exception:
                scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def calculate_accuracy(self, prediction, expected):
        """Implement your accuracy metric here."""
        pass
```

### 2. Batch Processing

```python
class BatchValidator(c.Vali):
    async def score_batch(self, modules: List[dict]) -> List[dict]:
        """Process multiple modules in parallel."""
        futures = []
        for module in modules:
            future = self.executor.submit(
                self.score_module,
                module,
                timeout=self.timeout
            )
            futures.append(future)
        
        results = []
        for future in c.as_completed(futures):
            try:
                result = await future
                results.append(result)
            except Exception as e:
                c.print(f"Batch error: {str(e)}")
        
        return results
```

### 3. State Management

```python
class StatefulValidator(c.Vali):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_state()
    
    def load_state(self):
        """Load validator state."""
        state = self.get('validator_state', {})
        self.historical_scores = state.get('scores', {})
        self.last_update = state.get('last_update', 0)
    
    def save_state(self):
        """Save validator state."""
        state = {
            'scores': self.historical_scores,
            'last_update': c.time()
        }
        self.put('validator_state', state)
    
    async def score(self, module) -> float:
        current_score = await super().score(module)
        
        # Update historical data
        module_key = module.get('key', '')
        if module_key:
            self.historical_scores[module_key] = {
                'score': current_score,
                'timestamp': c.time()
            }
            self.save_state()
        
        return current_score
```

## Network Integration

### 1. Voting Implementation

```python
class VotingValidator(c.Vali):
    def process_votes(self, results: List[dict]):
        """Process validation results for voting."""
        if not self.is_voting_network:
            return
        
        modules = []
        weights = []
        
        for result in results:
            if 'key' in result and 'score' in result:
                modules.append(result['key'])
                weights.append(result['score'])
        
        if modules and weights:
            return self.network_module.vote(
                modules=modules,
                weights=weights,
                key=self.key,
                subnet=self.subnet
            )
```

### 2. Subnet Configuration

```python
class SubnetValidator(c.Vali):
    def configure_subnet(self):
        """Configure validator for specific subnet."""
        subnet_info = self.network_module.get_subnet_info(
            self.subnet
        )
        
        self.min_stake = subnet_info.get('min_stake', 0)
        self.vote_interval = subnet_info.get('vote_interval', 100)
        self.reward_rate = subnet_info.get('reward_rate', 0.1)
```

## Testing Validators

```python
def test_validator():
    # Create test modules
    test_modules = [
        {
            'name': 'test_module_1',
            'key': 'key1',
            'address': 'localhost:8001'
        },
        {
            'name': 'test_module_2',
            'key': 'key2',
            'address': 'localhost:8002'
        }
    ]
    
    # Initialize validator
    validator = APIValidator(
        network='local',
        run_loop=False
    )
    
    # Test scoring
    results = validator.score_modules(test_modules)
    
    # Verify results
    assert len(results) == len(test_modules)
    for result in results:
        assert 'score' in result
        assert 0 <= result['score'] <= 1
```

## Best Practices

1. **Scoring Logic**
   - Use normalized scores (0.0 to 1.0)
   - Handle all exceptions gracefully
   - Include multiple evaluation criteria
   - Weight criteria appropriately

2. **Performance**
   - Implement efficient batch processing
   - Set appropriate timeouts
   - Cache results when possible
   - Monitor resource usage

3. **Security**
   - Validate all inputs
   - Protect against gaming attempts
   - Implement rate limiting
   - Secure sensitive data

4. **Maintenance**
   - Log validation results
   - Monitor validator health
   - Update test cases regularly
   - Track historical performance

## Common Issues

1. **Timeout Handling**
```python
async def safe_score(self, module) -> float:
    try:
        async with c.timeout(self.timeout):
            return await self.score(module)
    except c.TimeoutError:
        c.print(f"Timeout scoring module: {module.get('name')}")
        return 0.0
```

2. **Network Issues**
```python
def handle_network_error(self, error):
    if isinstance(error, c.ConnectionError):
        self.reconnect_count += 1
        if self.reconnect_count > self.max_retries:
            raise Exception("Max retries exceeded")
        c.sleep(self.retry_delay)
        return self.reconnect()
```

3. **State Corruption**
```python
def recover_state(self):
    try:
        self.load_state()
    except Exception:
        c.print("State corruption detected, resetting...")
        self.historical_scores = {}
        self.save_state()
```

## Next Steps

1. Study [Advanced Validation Patterns](22-Advanced-Validation.md)
2. Learn about [Network Security](43-Security-Best-Practices.md)
3. Explore [Token Economics](41-Token-Economics.md)

## Related Documentation

- [Module System](10-Module-System.md)
- [Network Architecture](11-Network-Architecture.md)
- [Blockchain Integration](12-Blockchain-Integration.md)
