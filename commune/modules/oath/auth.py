import commune as c
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import hashlib
import secrets

class Auth(c.Module):
    """
    Unified authentication and authorization system combining oath functionality.
    Handles API keys, tokens, permissions, rate limiting, and access control.
    """
    
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.tokens = {}
        self.api_keys = {}
        self.permissions = {}
        self.rate_limits = {}
        self.blacklist = set()
        self.whitelist = set()
        
    def generate_token(self, user_id: str, expiry_hours: int = 24, permissions: List[str] = None) -> str:
        """Generate a secure authentication token"""
        token = secrets.token_urlsafe(32)
        expiry = datetime.now() + timedelta(hours=expiry_hours)
        
        self.tokens[token] = {
            'user_id': user_id,
            'created': datetime.now().isoformat(),
            'expiry': expiry.isoformat(),
            'permissions': permissions or [],
            'active': True
        }
        
        return token
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify if token is valid and not expired"""
        if token not in self.tokens:
            return {'valid': False, 'reason': 'token_not_found'}
        
        token_data = self.tokens[token]
        
        if not token_data.get('active'):
            return {'valid': False, 'reason': 'token_inactive'}
        
        expiry = datetime.fromisoformat(token_data['expiry'])
        if datetime.now() > expiry:
            return {'valid': False, 'reason': 'token_expired'}
        
        return {'valid': True, 'user_id': token_data['user_id'], 'permissions': token_data['permissions']}
    
    def generate_api_key(self, user_id: str, name: str = None) -> str:
        """Generate API key for user"""
        api_key = f"sk_{secrets.token_urlsafe(32)}"
        
        self.api_keys[api_key] = {
            'user_id': user_id,
            'name': name or 'default',
            'created': datetime.now().isoformat(),
            'active': True,
            'usage_count': 0
        }
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Verify API key"""
        if api_key not in self.api_keys:
            return {'valid': False, 'reason': 'api_key_not_found'}
        
        key_data = self.api_keys[api_key]
        
        if not key_data.get('active'):
            return {'valid': False, 'reason': 'api_key_inactive'}
        
        key_data['usage_count'] += 1
        return {'valid': True, 'user_id': key_data['user_id']}
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has specific permission"""
        user_perms = self.permissions.get(user_id, [])
        return permission in user_perms or 'admin' in user_perms
    
    def grant_permission(self, user_id: str, permission: str) -> bool:
        """Grant permission to user"""
        if user_id not in self.permissions:
            self.permissions[user_id] = []
        
        if permission not in self.permissions[user_id]:
            self.permissions[user_id].append(permission)
        
        return True
    
    def revoke_permission(self, user_id: str, permission: str) -> bool:
        """Revoke permission from user"""
        if user_id in self.permissions and permission in self.permissions[user_id]:
            self.permissions[user_id].remove(permission)
            return True
        return False
    
    def check_rate_limit(self, user_id: str, limit: int = 100, window_minutes: int = 60) -> Dict[str, Any]:
        """Check if user is within rate limits"""
        now = datetime.now()
        
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []
        
        # Clean old requests outside window
        window_start = now - timedelta(minutes=window_minutes)
        self.rate_limits[user_id] = [
            req_time for req_time in self.rate_limits[user_id]
            if req_time > window_start
        ]
        
        current_count = len(self.rate_limits[user_id])
        
        if current_count >= limit:
            return {'allowed': False, 'count': current_count, 'limit': limit}
        
        self.rate_limits[user_id].append(now)
        return {'allowed': True, 'count': current_count + 1, 'limit': limit}
    
    def add_to_blacklist(self, identifier: str) -> bool:
        """Add user/IP to blacklist"""
        self.blacklist.add(identifier)
        return True
    
    def remove_from_blacklist(self, identifier: str) -> bool:
        """Remove from blacklist"""
        if identifier in self.blacklist:
            self.blacklist.remove(identifier)
            return True
        return False
    
    def is_blacklisted(self, identifier: str) -> bool:
        """Check if identifier is blacklisted"""
        return identifier in self.blacklist
    
    def add_to_whitelist(self, identifier: str) -> bool:
        """Add to whitelist"""
        self.whitelist.add(identifier)
        return True
    
    def is_whitelisted(self, identifier: str) -> bool:
        """Check if identifier is whitelisted"""
        return identifier in self.whitelist
    
    def authenticate(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Main authentication method - handles tokens and API keys"""
        # Check for blacklist
        user_id = credentials.get('user_id')
        if user_id and self.is_blacklisted(user_id):
            return {'authenticated': False, 'reason': 'blacklisted'}
        
        # Try token authentication
        if 'token' in credentials:
            result = self.verify_token(credentials['token'])
            if result['valid']:
                return {'authenticated': True, 'method': 'token', 'user_id': result['user_id']}
            return {'authenticated': False, 'reason': result['reason']}
        
        # Try API key authentication
        if 'api_key' in credentials:
            result = self.verify_api_key(credentials['api_key'])
            if result['valid']:
                return {'authenticated': True, 'method': 'api_key', 'user_id': result['user_id']}
            return {'authenticated': False, 'reason': result['reason']}
        
        return {'authenticated': False, 'reason': 'no_credentials'}
    
    def authorize(self, user_id: str, action: str, resource: str = None) -> Dict[str, Any]:
        """Check if user is authorized for action"""
        # Whitelist bypass
        if self.is_whitelisted(user_id):
            return {'authorized': True, 'reason': 'whitelisted'}
        
        # Check blacklist
        if self.is_blacklisted(user_id):
            return {'authorized': False, 'reason': 'blacklisted'}
        
        # Check rate limits
        rate_check = self.check_rate_limit(user_id)
        if not rate_check['allowed']:
            return {'authorized': False, 'reason': 'rate_limit_exceeded'}
        
        # Check permissions
        permission = f"{action}:{resource}" if resource else action
        if not self.check_permission(user_id, permission):
            return {'authorized': False, 'reason': 'insufficient_permissions'}
        
        return {'authorized': True}
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token"""
        if token in self.tokens:
            self.tokens[token]['active'] = False
            return True
        return False
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if api_key in self.api_keys:
            self.api_keys[api_key]['active'] = False
            return True
        return False
