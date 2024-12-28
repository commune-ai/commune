# Network Deployment Guide

This guide walks you through deploying Commune networks, from local development to production environments.

## Prerequisites
- Commune installed
- Basic understanding of networking concepts
- Docker (optional, for containerized deployment)
- Python 3.10+

## Deployment Types

### 1. Local Development Network

The simplest deployment for development and testing:

```python
import commune as c
from typing import Dict, Any, Optional
import asyncio

class MyModule(c.Module):
    """Example module for local deployment."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize module with optional configuration."""
        super().__init__()
        self.config = config or {}
    
    async def process(self, data: Any) -> str:
        """Process input data.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data string
        """
        return f"Processed: {data}"
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get module statistics.
        
        Returns:
            Dictionary of module stats
        """
        return {
            'processed_count': self.processed_count,
            'uptime': self.get_uptime(),
            'errors': self.error_count
        }

async def run_local_server():
    """Start local development server."""
    try:
        # Create and serve module
        module = MyModule()
        server = await c.serve(
            module,
            network='local',
            port=8000,
            host='0.0.0.0'
        )
        print("Local server running on port 8000")
        
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        await server.stop()

async def test_connection():
    """Test connection to local server."""
    try:
        # Connect to module
        client = await c.connect('my_module')
        
        # Test processing
        result = await client.process("test data")
        print(f"Result: {result}")
        
        # Get stats
        stats = await client.get_stats()
        print(f"Module stats: {stats}")
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == '__main__':
    # Run server
    asyncio.run(run_local_server())
```

### 2. Production Network

For production environments with multiple nodes:

```python
import commune as c
from typing import Dict, Any, List, Optional
import yaml
import asyncio
from dataclasses import dataclass

@dataclass
class NetworkConfig:
    """Configuration for production network."""
    name: str
    subnet: str
    tempo: int
    validators: int
    min_stake: float
    block_time: int

@dataclass
class SecurityConfig:
    """Security configuration for network."""
    key_type: str
    authentication: bool
    rate_limit: bool
    max_requests: int

@dataclass
class ProductionConfig:
    """Complete production configuration."""
    network: NetworkConfig
    security: SecurityConfig

class ProductionNetwork:
    """Manager for production network deployment."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration file.
        
        Args:
            config_path: Path to YAML config file
        """
        self.config = self._load_config(config_path)
        self.modules: Dict[str, c.Module] = {}
    
    @staticmethod
    def _load_config(path: str) -> ProductionConfig:
        """Load configuration from YAML file.
        
        Args:
            path: Path to config file
            
        Returns:
            Parsed configuration object
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
            return ProductionConfig(
                network=NetworkConfig(**config_dict['network']),
                security=SecurityConfig(**config_dict['security'])
            )
    
    async def start(self) -> None:
        """Start the production network."""
        try:
            # Initialize network
            network = await c.network(
                network=self.config.network.name,
                subnet=self.config.network.subnet,
                validators=self.config.network.validators,
                min_stake=self.config.network.min_stake,
                block_time=self.config.network.block_time
            )
            
            # Apply security settings
            await network.configure_security(
                key_type=self.config.security.key_type,
                authentication=self.config.security.authentication,
                rate_limit=self.config.security.rate_limit,
                max_requests=self.config.security.max_requests
            )
            
            print(f"Production network {self.config.network.name} started")
            
            # Keep network running
            while True:
                await self._monitor_network(network)
                await asyncio.sleep(self.config.network.tempo)
        except Exception as e:
            print(f"Network error: {e}")
            raise
    
    async def _monitor_network(
        self,
        network: c.Module
    ) -> None:
        """Monitor network health.
        
        Args:
            network: Network module to monitor
        """
        try:
            stats = await network.get_stats()
            print(f"Network stats: {stats}")
            
            # Check validator health
            validators = await network.get_validators()
            for v in validators:
                if v['score'] < 0.5:
                    print(f"Warning: Low validator score - {v['id']}")
        except Exception as e:
            print(f"Monitoring error: {e}")

class LoadBalancer:
    """Load balancer for distributing requests."""
    
    def __init__(
        self,
        strategy: str = 'round_robin',
        check_interval: int = 30
    ):
        """Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy
            check_interval: Health check interval in seconds
        """
        self.strategy = strategy
        self.check_interval = check_interval
        self.modules: List[c.Module] = []
        self.current_index = 0
    
    async def add_module(
        self,
        module: c.Module
    ) -> None:
        """Add module to load balancer.
        
        Args:
            module: Module to add
        """
        self.modules.append(module)
    
    async def get_next_module(self) -> Optional[c.Module]:
        """Get next available module based on strategy.
        
        Returns:
            Next module to use or None if none available
        """
        if not self.modules:
            return None
            
        if self.strategy == 'round_robin':
            module = self.modules[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.modules)
            return module
        
        return self.modules[0]  # Default to first module
    
    async def check_health(self) -> None:
        """Check health of all modules."""
        unhealthy = []
        for module in self.modules:
            try:
                await module.ping()
            except Exception:
                unhealthy.append(module)
        
        # Remove unhealthy modules
        for module in unhealthy:
            self.modules.remove(module)
            print(f"Removed unhealthy module: {module}")

async def main():
    """Run production deployment."""
    try:
        # Start production network
        network = ProductionNetwork('config.yaml')
        await network.start()
    except KeyboardInterrupt:
        print("\nShutting down production network...")
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == '__main__':
    asyncio.run(main())
```

### 3. Docker Deployment

For containerized environments:

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
ENV PYTHONPATH=/app

CMD ["python", "deployment.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  validator1:
    build: .
    environment:
      - NODE_TYPE=validator
      - NODE_KEY=${VALIDATOR1_KEY}
    ports:
      - "8001:8001"
    volumes:
      - validator1_data:/app/data

  validator2:
    build: .
    environment:
      - NODE_TYPE=validator
      - NODE_KEY=${VALIDATOR2_KEY}
    ports:
      - "8002:8002"
    volumes:
      - validator2_data:/app/data

  api_node:
    build: .
    environment:
      - NODE_TYPE=api
      - NODE_KEY=${API_KEY}
    ports:
      - "8000:8000"
    depends_on:
      - validator1
      - validator2

volumes:
  validator1_data:
  validator2_data:
```

## Network Configuration

### 1. Basic Network Setup

```python
def setup_network(network_type='local'):
    """Configure network based on type."""
    if network_type == 'local':
        config = {
            'network': 'local',
            'tempo': 30,
            'authentication': False
        }
    elif network_type == 'testnet':
        config = {
            'network': 'subspace/test',
            'tempo': 60,
            'authentication': True
        }
    else:  # mainnet
        config = {
            'network': 'subspace/main',
            'tempo': 120,
            'authentication': True
        }
    
    return c.module('network')(**config)
```

### 2. Security Configuration

```python
def configure_security(server, security_level='high'):
    """Configure server security settings."""
    if security_level == 'high':
        server.update_config({
            'rate_limit': True,
            'max_requests': 100,
            'request_timeout': 30,
            'authentication': True,
            'ssl_enabled': True
        })
    elif security_level == 'medium':
        server.update_config({
            'rate_limit': True,
            'max_requests': 1000,
            'request_timeout': 60,
            'authentication': True,
            'ssl_enabled': False
        })
    else:  # development
        server.update_config({
            'rate_limit': False,
            'authentication': False,
            'ssl_enabled': False
        })
```

## Scaling Strategies

### 1. Horizontal Scaling

```python
def scale_horizontally(module_name, replicas=3):
    """Scale module horizontally."""
    base_port = 8000
    processes = []
    
    for i in range(replicas):
        port = base_port + i
        process = c.serve(
            module_name,
            port=port,
            network='subspace',
            load_balancing=True
        )
        processes.append(process)
    
    return processes
```

### 2. Load Balancing

```python
def setup_load_balancer(modules):
    """Configure load balancing for modules."""
    return c.module('loadbalancer')(
        modules=modules,
        strategy='round_robin',
        health_check=True,
        check_interval=30
    )
```

## Monitoring and Maintenance

### 1. Health Checks

```python
async def monitor_network_health(network):
    """Monitor network health."""
    while True:
        try:
            # Check module status
            modules = network.modules()
            for module in modules:
                status = await c.acall(
                    f"{module['name']}/health"
                )
                if not status['healthy']:
                    c.print(f"Module {module['name']} unhealthy")
            
            # Check network metrics
            metrics = network.metrics()
            if metrics['latency'] > 1.0:
                c.print("High network latency detected")
                
            await c.sleep(60)
        except Exception as e:
            c.print(f"Monitoring error: {str(e)}")
```

### 2. Backup and Recovery

```python
def backup_network_state(network, backup_path):
    """Backup network state."""
    state = {
        'modules': network.modules(),
        'config': network.params(),
        'timestamp': c.time()
    }
    c.put_json(backup_path, state)
    return state

def restore_network_state(backup_path):
    """Restore network from backup."""
    state = c.get_json(backup_path)
    network = c.module('network')(
        network=state['config']['network']
    )
    
    for module in state['modules']:
        c.serve(
            module['name'],
            port=module['port'],
            network=state['config']['network']
        )
    
    return network
```

## Common Issues and Solutions

### 1. Network Connectivity

```python
def troubleshoot_connection(module_name):
    """Troubleshoot network connectivity."""
    try:
        # Check if module is running
        if not c.module_exists(module_name):
            return "Module not running"
        
        # Test connection
        client = c.connect(module_name)
        response = client.health()
        
        if response['status'] != 'ok':
            return f"Module unhealthy: {response}"
        
        return "Connection OK"
    except Exception as e:
        return f"Connection error: {str(e)}"
```

### 2. Performance Issues

```python
def optimize_performance(network):
    """Optimize network performance."""
    # Adjust batch size based on load
    current_load = network.metrics()['load']
    if current_load > 0.8:
        network.update_config({
            'batch_size': 64,
            'max_workers': c.cpu_count() * 2
        })
    else:
        network.update_config({
            'batch_size': 128,
            'max_workers': c.cpu_count() * 4
        })
```

## Best Practices

1. **Network Planning**
   - Start with local development network
   - Test thoroughly on testnet
   - Use staging environment before mainnet
   - Document network topology

2. **Security**
   - Use strong authentication
   - Enable rate limiting
   - Implement SSL/TLS
   - Regular security audits
   - Monitor for suspicious activity

3. **Performance**
   - Monitor resource usage
   - Use appropriate batch sizes
   - Implement caching
   - Regular performance testing
   - Scale based on metrics

4. **Maintenance**
   - Regular backups
   - Automated health checks
   - Update documentation
   - Monitor logs
   - Plan for disaster recovery

## Next Steps

1. Study [Advanced Network Features](23-Advanced-Network.md)
2. Learn about [Security Best Practices](24-Security-Guide.md)
3. Explore [Performance Optimization](25-Performance-Guide.md)

## Related Documentation

- [Network Architecture](11-Network-Architecture.md)
- [Module System](10-Module-System.md)
- [Blockchain Integration](12-Blockchain-Integration.md)
