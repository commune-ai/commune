# Network Architecture

Commune's network architecture is designed for distributed computing with built-in security, scalability, and fault tolerance. This document explains the core components and their interactions.

## Core Components

### 1. Server

The Server component is the backbone of Commune's network architecture, providing:

```python
import commune as c
from typing import Optional, Dict, Any

class MyServer(c.Module):
    """Custom server module with configurable settings."""
    
    def __init__(
        self,
        port: int = 8000,
        host: str = "0.0.0.0",
        network: str = "subspace",
        max_workers: int = 4
    ):
        super().__init__()
        self.server = c.server(
            module=self,
            port=port,
            host=host,
            network=network,
            max_workers=max_workers
        )
    
    async def start(self) -> None:
        """Start the server."""
        await self.server.start()
    
    async def stop(self) -> None:
        """Stop the server gracefully."""
        await self.server.stop()
    
    async def handle_request(
        self,
        method: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle incoming requests."""
        try:
            handler = getattr(self, method)
            result = await handler(**data)
            return {
                'success': True,
                'result': result
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Example usage
async def main():
    server = MyServer(port=8000)
    await server.start()
```

Key features:
- FastAPI-based HTTP server
- Request size limiting
- CORS support
- SSE (Server-Sent Events) capability
- Rate limiting
- User authentication

### 2. Network Manager

The Network component handles module discovery and communication:

```python
import commune as c
from typing import List, Dict, Optional

class NetworkManager:
    """Manages network connections and module discovery."""
    
    def __init__(
        self,
        network: str = 'local',
        tempo: int = 60,
        discovery_port: int = 8888
    ):
        self.network = c.network(
            network=network,
            tempo=tempo,
            discovery_port=discovery_port
        )
    
    async def discover_modules(
        self,
        module_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Discover available modules on the network."""
        modules = await self.network.modules()
        
        if module_type:
            return [
                m for m in modules
                if m.get('type') == module_type
            ]
        return modules
    
    async def get_module_info(
        self,
        module_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get detailed information about a module."""
        try:
            return await self.network.get_module_info(module_name)
        except Exception:
            return None
    
    async def connect(
        self,
        module_name: str,
        timeout: float = 5.0
    ) -> Optional[c.Module]:
        """Connect to a remote module."""
        try:
            return await self.network.connect(
                module_name,
                timeout=timeout
            )
        except TimeoutError:
            return None

# Example usage
async def main():
    manager = NetworkManager(network='testnet')
    modules = await manager.discover_modules()
    print(f"Found {len(modules)} modules")
    
    # Connect to specific module
    module = await manager.connect('storage')
    if module:
        info = await manager.get_module_info('storage')
        print(f"Connected to storage module: {info}")
```

## Network Types

### 1. Local Network
```python
import commune as c
from typing import List, Dict

async def setup_local_network(
    modules: List[Dict[str, Any]]
) -> None:
    """Set up a local development network."""
    network = c.network(network='local')
    
    # Start modules
    for module_config in modules:
        module = await network.start_module(
            name=module_config['name'],
            config=module_config.get('config', {})
        )
        print(f"Started {module_config['name']}")
    
    # Wait for network stabilization
    await network.wait_ready()

# Example configuration
modules_config = [
    {
        'name': 'storage',
        'config': {'path': '/data'}
    },
    {
        'name': 'compute',
        'config': {'workers': 2}
    }
]

async def main():
    await setup_local_network(modules_config)
```

### 2. Production Network
```python
import commune as c
from typing import Dict, Any
import yaml

def load_network_config(path: str) -> Dict[str, Any]:
    """Load network configuration from YAML."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

async def setup_production_network(
    config_path: str
) -> None:
    """Set up a production network with config."""
    config = load_network_config(config_path)
    
    network = c.network(
        network=config['network']['name'],
        subnet=config['network']['subnet'],
        validators=config['network']['validators'],
        min_stake=config['network']['min_stake']
    )
    
    # Apply security settings
    network.set_security_config(
        key_type=config['security']['key_type'],
        authentication=config['security']['authentication'],
        rate_limit=config['security']['rate_limit']
    )
    
    await network.start()

# Example usage
async def main():
    await setup_production_network('network_config.yaml')
```

## Best Practices

### 1. Error Handling
```python
import commune as c
from typing import Optional, Dict, Any

class NetworkModule(c.Module):
    """Network-aware module with error handling."""
    
    async def safe_remote_call(
        self,
        module: str,
        method: str,
        *args,
        timeout: float = 5.0,
        retries: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """Make safe remote calls with retries."""
        for attempt in range(retries):
            try:
                remote = await c.connect(
                    module,
                    timeout=timeout
                )
                result = await getattr(remote, method)(
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
                continue

# Example usage
async def main():
    module = NetworkModule()
    result = await module.safe_remote_call(
        'storage',
        'get_data',
        key='test'
    )
    print(result)
```

## Communication Patterns

### 1. Direct Calls
```python
# Synchronous call
result = c.call('module/function', data)

# Async call
result = await c.acall('module/function', data)
```

### 2. Event-Based
```python
# Subscribe to events
async for event in c.subscribe('module/events'):
    process_event(event)
```

### 3. Broadcast
```python
# Send to all modules
results = c.broadcast('function', data)
```

## Security Model

### 1. Authentication
```python
# Key-based authentication
server = c.server(
    module=module,
    key='my_key',
    crypto_type='sr25519'
)
```

### 2. Rate Limiting
```python
# Configure rate limits
server.rates = {
    'max': 10,       # Default max requests/second
    'local': 10000,  # Local network limit
    'stake': 1000,   # Staked users limit
    'owner': 10000   # Owner limit
}
```

### 3. Access Control
```python
class SecureModule(c.Module):
    def __init__(self):
        self.require_stake = True
        self.min_stake = 100
```

## Network Configuration

### 1. Basic Setup
```python
network_config = {
    'network': 'local',
    'tempo': 60,
    'blocktime': 8,
    'min_stake': 0
}

network = c.network(**network_config)
```

### 2. Advanced Configuration
```python
server_config = {
    'max_request_staleness': 4,
    'max_network_staleness': 60,
    'user_data_lifetime': 3600,
    'multipliers': {
        'stake': 1,
        'stake_to': 1,
        'stake_from': 1
    }
}
```

## Load Balancing

### 1. Module Discovery
```python
# Find available modules
modules = network.modules(
    search='service_name',
    max_age=60,
    features=['name', 'address', 'key']
)
```

### 2. Health Checks
```python
# Check module health
status = c.health_check('module_name')

# Get network status
network_status = network.status()
```

## Fault Tolerance

### 1. Automatic Retries
```python
result = c.call(
    'module/function',
    data,
    retries=3,
    retry_delay=1
)
```

### 2. Failover
```python
# Configure backup modules
c.set_failover(['module1', 'module2', 'module3'])
```

## Performance Optimization

### 1. Caching
```python
# Enable module caching
modules = network.modules(
    max_age=60,
    update=False
)
```

### 2. Connection Pooling
```python
# Configure connection pool
server = c.server(
    max_connections=100,
    keep_alive=True
)
```

## Monitoring and Debugging

### 1. Logging
```python
# Enable detailed logging
c.set_log_level('debug')

# Monitor network events
c.monitor_network()
```

### 2. Metrics
```python
# Get network metrics
metrics = network.metrics()

# Get server stats
stats = server.stats()
```

## Common Issues

1. **Connection Problems**
```python
# Check network connectivity
c.check_connection()

# Reset network state
c.reset_network()
```

2. **Performance Issues**
```python
# Monitor latency
latency = c.measure_latency('module_name')

# Check resource usage
usage = c.resource_usage()
```

## Next Steps

1. Learn about [Blockchain Integration](12-Blockchain-Integration.md)
2. Explore [Custom Validators](40-Custom-Validators.md)
3. Study [Security Best Practices](43-Security-Best-Practices.md)
