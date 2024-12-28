# Security Best Practices

This guide outlines essential security practices for building and deploying Commune applications.

## Core Security Principles

1. **Defense in Depth**
   - Multiple layers of security
   - No single point of failure
   - Comprehensive monitoring

2. **Least Privilege**
   - Minimal access rights
   - Role-based access control
   - Regular access reviews

3. **Secure by Default**
   - Conservative default settings
   - Explicit security configurations
   - Fail-safe defaults

## Authentication and Authorization

### 1. Key-Based Authentication

```python
import commune as c

class SecureModule(c.Module):
    def __init__(self, key=None):
        super().__init__()
        self.server = c.server(
            module=self,
            key=key,
            authentication=True
        )
        
    def secure_endpoint(self, data, key=None):
        """Endpoint requiring authentication."""
        if not self.verify_key(key):
            raise c.AuthError("Invalid key")
        return self.process_data(data)
```

### 2. Role-Based Access Control

```python
class RBACModule(c.Module):
    def __init__(self):
        super().__init__()
        self.user_manager = c.module('user')
    
    def setup_roles(self):
        """Initialize role-based permissions."""
        roles = {
            'admin': ['read', 'write', 'delete', 'manage'],
            'user': ['read', 'write'],
            'viewer': ['read']
        }
        for role, permissions in roles.items():
            self.user_manager.add_role(role, permissions)
    
    async def protected_endpoint(self, user_key, action):
        """Check permissions before execution."""
        user = self.user_manager.get_user(user_key)
        if not user:
            raise c.AuthError("User not found")
        
        if not self.user_manager.has_permission(
            user['role'], action
        ):
            raise c.AuthError("Insufficient permissions")
        
        return await self.execute_action(action)
```

### 3. Token Management

```python
class TokenManager(c.Module):
    def __init__(self):
        super().__init__()
        self.token_lifetime = 3600  # 1 hour
    
    def generate_token(self, user_key):
        """Generate a secure session token."""
        token = c.generate_random_string(32)
        expiry = c.time() + self.token_lifetime
        
        token_data = {
            'token': token,
            'user': user_key,
            'expiry': expiry
        }
        
        # Store encrypted token data
        self.put(
            f'token_{token}',
            token_data,
            encrypt=True
        )
        
        return token
    
    def verify_token(self, token):
        """Verify token validity."""
        token_data = self.get(f'token_{token}')
        if not token_data:
            return False
        
        if token_data['expiry'] < c.time():
            self.revoke_token(token)
            return False
        
        return token_data
```

## Network Security

### 1. SSL/TLS Configuration

```python
def configure_ssl(server):
    """Configure SSL/TLS for server."""
    ssl_config = {
        'cert_path': '/path/to/cert.pem',
        'key_path': '/path/to/key.pem',
        'protocols': ['TLSv1.2', 'TLSv1.3'],
        'ciphers': 'ECDHE-ECDSA-AES128-GCM-SHA256'
    }
    
    server.update_config({
        'ssl_enabled': True,
        'ssl_config': ssl_config
    })
```

### 2. Rate Limiting

```python
class RateLimitedModule(c.Module):
    def __init__(self):
        super().__init__()
        self.rate_limits = {
            'default': 100,  # requests per minute
            'admin': 1000,
            'api': 500
        }
    
    def configure_rate_limits(self):
        """Set up rate limiting."""
        for role, limit in self.rate_limits.items():
            self.server.add_rate_limit(
                role=role,
                limit=limit,
                window=60  # seconds
            )
    
    async def rate_limited_endpoint(self, user_key):
        """Endpoint with rate limiting."""
        user = self.get_user(user_key)
        role = user.get('role', 'default')
        
        if not self.check_rate_limit(user_key, role):
            raise c.RateLimitError(
                f"Rate limit exceeded for {role}"
            )
        
        return await self.process_request()
```

### 3. DDoS Protection

```python
class DDoSProtectedModule(c.Module):
    def __init__(self):
        super().__init__()
        self.setup_ddos_protection()
    
    def setup_ddos_protection(self):
        """Configure DDoS protection."""
        protection = {
            'max_connections': 1000,
            'connection_timeout': 30,
            'blacklist_threshold': 100,
            'whitelist': ['trusted_ip_1', 'trusted_ip_2']
        }
        
        self.server.update_config({
            'ddos_protection': protection
        })
    
    def monitor_attacks(self):
        """Monitor for potential DDoS attacks."""
        metrics = self.server.metrics()
        if metrics['requests_per_second'] > 1000:
            self.trigger_ddos_mitigation()
```

## Data Security

### 1. Encryption

```python
class SecureStorage(c.Module):
    def __init__(self):
        super().__init__()
        self.setup_encryption()
    
    def setup_encryption(self):
        """Configure encryption settings."""
        self.encryption_config = {
            'algorithm': 'AES-256-GCM',
            'key_derivation': 'PBKDF2',
            'iterations': 100000
        }
    
    def store_sensitive_data(self, data, password):
        """Store encrypted data."""
        encrypted = c.encrypt(
            data,
            password=password,
            **self.encryption_config
        )
        return self.put('sensitive_data', encrypted)
    
    def retrieve_sensitive_data(self, password):
        """Retrieve and decrypt data."""
        encrypted = self.get('sensitive_data')
        return c.decrypt(
            encrypted,
            password=password
        )
```

### 2. Secure Configuration

```python
class SecureConfig(c.Module):
    def __init__(self):
        super().__init__()
        self.load_secure_config()
    
    def load_secure_config(self):
        """Load configuration securely."""
        # Never store sensitive data in code
        self.config = {
            'api_key': c.get_env('API_KEY'),
            'db_password': c.get_env('DB_PASSWORD'),
            'secret_key': c.get_env('SECRET_KEY')
        }
        
        # Validate configuration
        self.validate_config()
    
    def validate_config(self):
        """Ensure all required secure configs exist."""
        required = ['api_key', 'secret_key']
        missing = [k for k in required 
                  if not self.config.get(k)]
        
        if missing:
            raise c.ConfigError(
                f"Missing secure configs: {missing}"
            )
```

## Monitoring and Auditing

### 1. Security Logging

```python
class SecureLogger(c.Module):
    def __init__(self):
        super().__init__()
        self.setup_logging()
    
    def setup_logging(self):
        """Configure secure logging."""
        self.log_config = {
            'level': 'INFO',
            'format': (
                '%(asctime)s [%(levelname)s] '
                '%(message)s [%(user)s]'
            ),
            'sensitive_fields': [
                'password', 'token', 'key'
            ]
        }
    
    def log_security_event(self, event_type, user, **data):
        """Log security events safely."""
        # Remove sensitive data
        safe_data = self.sanitize_data(data)
        
        # Add context
        context = {
            'user': user,
            'ip': c.get_client_ip(),
            'timestamp': c.time()
        }
        
        self.log(event_type, safe_data, context)
```

### 2. Audit Trail

```python
class AuditTrail(c.Module):
    def __init__(self):
        super().__init__()
        self.setup_audit()
    
    def setup_audit(self):
        """Configure audit trail."""
        self.audit_config = {
            'storage': 'encrypted',
            'retention': 90,  # days
            'events': [
                'login', 'logout', 'data_access',
                'config_change', 'permission_change'
            ]
        }
    
    def record_audit(self, event, user, data):
        """Record audit event."""
        audit_entry = {
            'event': event,
            'user': user,
            'timestamp': c.time(),
            'data': data,
            'hash': self.hash_audit_data(data)
        }
        
        self.store_audit(audit_entry)
```

## Best Practices

1. **Key Management**
   - Rotate keys regularly
   - Use strong key derivation
   - Secure key storage
   - Never hardcode keys

2. **Network Security**
   - Enable SSL/TLS
   - Implement rate limiting
   - Use IP whitelisting
   - Monitor traffic patterns

3. **Data Protection**
   - Encrypt sensitive data
   - Sanitize user input
   - Regular backups
   - Secure data deletion

4. **Access Control**
   - Implement RBAC
   - Regular access reviews
   - Strong authentication
   - Session management

5. **Monitoring**
   - Security logging
   - Audit trails
   - Alerting system
   - Regular reviews

## Common Vulnerabilities

1. **Input Validation**
```python
def validate_input(data):
    """Validate and sanitize input."""
    if not isinstance(data, dict):
        raise ValueError("Invalid input format")
    
    # Sanitize known fields
    for field in ['name', 'description']:
        if field in data:
            data[field] = c.sanitize_string(
                data[field]
            )
    
    return data
```

2. **SQL Injection Prevention**
```python
def safe_query(self, table, conditions):
    """Prevent SQL injection."""
    # Use parameterized queries
    query = "SELECT * FROM ? WHERE ?"
    params = [table, conditions]
    
    return self.db.execute(query, params)
```

3. **XSS Prevention**
```python
def render_user_content(content):
    """Prevent XSS in user content."""
    return c.escape_html(content)
```

## Security Checklist

1. **Authentication**
   - [ ] Implement key-based auth
   - [ ] Set up role-based access
   - [ ] Configure session management
   - [ ] Enable MFA where possible

2. **Network**
   - [ ] Enable SSL/TLS
   - [ ] Configure firewalls
   - [ ] Set up rate limiting
   - [ ] Implement DDoS protection

3. **Data**
   - [ ] Encrypt sensitive data
   - [ ] Secure configuration
   - [ ] Regular backups
   - [ ] Data sanitization

4. **Monitoring**
   - [ ] Security logging
   - [ ] Audit trail
   - [ ] Alerting system
   - [ ] Regular reviews

## Next Steps

1. Study [Advanced Security Patterns](24-Advanced-Security.md)
2. Learn about [Compliance](25-Compliance-Guide.md)
3. Explore [Security Testing](26-Security-Testing.md)

## Related Documentation

- [Network Architecture](11-Network-Architecture.md)
- [Network Deployment](22-Network-Deployment.md)
- [Module System](10-Module-System.md)
