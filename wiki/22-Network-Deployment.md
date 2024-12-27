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

# Create and serve a module
class MyModule(c.Module):
    def process(self, data):
        return f"Processed: {data}"

# Start local server
c.serve('my_module', network='local', port=8000)

# Connect from another process
client = c.connect('my_module')
result = client.process("test data")
```

### 2. Production Network

For production environments with multiple nodes:

```python
# config.yaml
network:
  name: "prod_network"
  subnet: "main"
  tempo: 60
  validators: 5
  min_stake: 1000
  block_time: 8

security:
  key_type: "sr25519"
  authentication: true
  rate_limit: true
  max_requests: 1000

modules:
  - name: "api_module"
    port: 8001
    replicas: 3
  - name: "storage_module"
    port: 8002
    replicas: 2
```

```python
# deployment.py
import commune as c
import yaml

def deploy_network():
    # Load configuration
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Initialize network
    network = c.module('network')(
        network=config['network']['name'],
        subnet=config['network']['subnet'],
        tempo=config['network']['tempo']
    )
    
    # Deploy modules
    for module_config in config['modules']:
        for i in range(module_config['replicas']):
            port = module_config['port'] + i
            c.serve(
                module_config['name'],
                port=port,
                network=config['network']['name'],
                key_type=config['security']['key_type'],
                authentication=config['security']['authentication']
            )

if __name__ == "__main__":
    deploy_network()
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
