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
from typing import Optional, Union, List, Dict

class APIValidator(c.Vali):
    def __init__(self,
                network='local',
                subnet: Optional[str] = None,
                batch_size: int = 32,
                timeout: int = 5):
        
        super().__init__(
            network=network,
            subnet=subnet,
            batch_size=batch_size,
            timeout=timeout,
            score=self.score_api,  # Set our custom scoring function
            run_loop=True  # Auto-start validation loop
        )
        
        # Initialize validator-specific attributes
        self.required_endpoints = ['info', 'status', 'health']
        self.min_response_time = 0.1
        self.max_response_time = 2.0
    
    async def score_api(self, module) -> float:
        """Score an API module based on:
        1. Endpoint availability
        2. Response time
        3. Error handling
        4. Data quality
        """
        try:
            # Check basic module info
            info = await module.info()
            if not isinstance(info, dict):
                return 0.0
            
            # Test required endpoints
            endpoint_scores = []
            for endpoint in self.required_endpoints:
                if endpoint in info.get('endpoints', []):
                    endpoint_scores.append(1.0)
                else:
                    endpoint_scores.append(0.0)
            
            # Test response time
            start_time = c.time()
            await module.health()
            response_time = c.time() - start_time
            
            # Calculate time score
            time_score = 1.0
            if response_time > self.max_response_time:
                time_score = 0.0
            elif response_time > self.min_response_time:
                time_score = 1.0 - ((response_time - self.min_response_time) 
                                  / (self.max_response_time - self.min_response_time))
            
            # Combine scores
            endpoint_score = sum(endpoint_scores) / len(self.required_endpoints)
            final_score = (endpoint_score * 0.7) + (time_score * 0.3)
            
            return final_score
            
        except Exception as e:
            c.print(f"Scoring error: {str(e)}")
            return 0.0
```

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
