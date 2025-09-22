#!/usr/bin/env python3
"""
MCP Utils - Utility functions and helpers for the MCP system

Enhanced utilities for the commune MCP architecture.
Built with Leonardo da Vinci's principle: 'Simplicity is the ultimate sophistication'
"""

import asyncio
import json
import logging
import time
import functools
import inspect
from typing import Any, Dict, List, Optional, Callable, TypeVar, Union
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import traceback

# Type definitions
T = TypeVar('T')
JSON = Dict[str, Any]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Base exception for MCP-related errors."""
    pass


class ToolError(MCPError):
    """Exception raised when a tool execution fails."""
    pass


class ValidationError(MCPError):
    """Exception raised when validation fails."""
    pass


# Decorators
def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry decorator with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"Max retries ({max_attempts}) reached for {func.__name__}")
                        raise
                    logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}. Retrying in {current_delay}s...")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"Max retries ({max_attempts}) reached for {func.__name__}")
                        raise
                    logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def cache_result(ttl: int = 300):
    """Cache function results with TTL (time-to-live)."""
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str((args, tuple(sorted(kwargs.items()))))
            now = time.time()
            
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            
            # Clean expired entries
            expired = [k for k, (_, ts) in cache.items() if now - ts >= ttl]
            for k in expired:
                del cache[k]
            
            return result
        return wrapper
    return decorator


def validate_params(**validators):
    """Validate function parameters."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValidationError(f"Validation failed for parameter '{param_name}' with value '{value}'")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def measure_time(func: Callable) -> Callable:
    """Measure and log function execution time."""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


# Context Managers
@contextmanager
def error_handler(operation: str = "operation", raise_on_error: bool = True):
    """Context manager for handling errors gracefully."""
    try:
        yield
    except Exception as e:
        logger.error(f"Error during {operation}: {e}")
        logger.debug(traceback.format_exc())
        if raise_on_error:
            raise


@contextmanager
def timer(name: str = "operation"):
    """Context manager for timing operations."""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"{name} took {elapsed:.3f}s")


# Utility Classes
class MessageQueue:
    """Simple async message queue for inter-module communication."""
    
    def __init__(self, maxsize: int = 0):
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.subscribers = []
    
    async def put(self, message: Any):
        """Put a message in the queue."""
        await self.queue.put(message)
        for subscriber in self.subscribers:
            await subscriber(message)
    
    async def get(self) -> Any:
        """Get a message from the queue."""
        return await self.queue.get()
    
    def subscribe(self, callback: Callable):
        """Subscribe to queue messages."""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from queue messages."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)


class Registry:
    """Registry for managing modules and tools."""
    
    def __init__(self):
        self._items = {}
        self._metadata = {}
    
    def register(self, name: str, item: Any, metadata: Optional[Dict] = None):
        """Register an item with optional metadata."""
        self._items[name] = item
        if metadata:
            self._metadata[name] = metadata
        logger.info(f"Registered {name}")
    
    def unregister(self, name: str):
        """Unregister an item."""
        if name in self._items:
            del self._items[name]
            if name in self._metadata:
                del self._metadata[name]
            logger.info(f"Unregistered {name}")
    
    def get(self, name: str) -> Optional[Any]:
        """Get a registered item."""
        return self._items.get(name)
    
    def get_metadata(self, name: str) -> Optional[Dict]:
        """Get metadata for a registered item."""
        return self._metadata.get(name)
    
    def list(self) -> List[str]:
        """List all registered items."""
        return list(self._items.keys())
    
    def search(self, query: str) -> List[str]:
        """Search for items matching a query."""
        query_lower = query.lower()
        matches = []
        for name in self._items:
            if query_lower in name.lower():
                matches.append(name)
            elif name in self._metadata:
                meta = self._metadata[name]
                if 'description' in meta and query_lower in meta['description'].lower():
                    matches.append(name)
                elif 'tags' in meta and any(query_lower in tag.lower() for tag in meta['tags']):
                    matches.append(name)
        return matches


class StateManager:
    """Manage state for modules and agents."""
    
    def __init__(self, persist_path: Optional[Path] = None):
        self._state = {}
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self.load()
    
    def set(self, key: str, value: Any, namespace: str = 'default'):
        """Set a state value."""
        if namespace not in self._state:
            self._state[namespace] = {}
        self._state[namespace][key] = value
        if self._persist_path:
            self.save()
    
    def get(self, key: str, namespace: str = 'default', default: Any = None) -> Any:
        """Get a state value."""
        if namespace in self._state and key in self._state[namespace]:
            return self._state[namespace][key]
        return default
    
    def update(self, updates: Dict[str, Any], namespace: str = 'default'):
        """Update multiple state values."""
        if namespace not in self._state:
            self._state[namespace] = {}
        self._state[namespace].update(updates)
        if self._persist_path:
            self.save()
    
    def clear(self, namespace: Optional[str] = None):
        """Clear state for a namespace or all state."""
        if namespace:
            if namespace in self._state:
                del self._state[namespace]
        else:
            self._state.clear()
        if self._persist_path:
            self.save()
    
    def save(self):
        """Save state to disk."""
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, 'w') as f:
                json.dump(self._state, f, indent=2, default=str)
    
    def load(self):
        """Load state from disk."""
        if self._persist_path and self._persist_path.exists():
            with open(self._persist_path, 'r') as f:
                self._state = json.load(f)


# Helper Functions
def parse_json_safe(text: str, default: Any = None) -> Any:
    """Safely parse JSON with fallback."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse JSON: {text[:100]}...")
        return default


def format_timestamp(dt: Optional[datetime] = None, fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
    """Format a datetime object or current time."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime(fmt)


def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def merge_dicts(dict1: Dict, dict2: Dict, deep: bool = True) -> Dict:
    """Merge two dictionaries."""
    result = dict1.copy()
    if deep:
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value, deep=True)
            else:
                result[key] = value
    else:
        result.update(dict2)
    return result


def batch_process(items: List[T], batch_size: int = 10) -> List[List[T]]:
    """Split items into batches."""
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


async def run_parallel(*coroutines, return_exceptions: bool = False) -> List[Any]:
    """Run multiple coroutines in parallel."""
    return await asyncio.gather(*coroutines, return_exceptions=return_exceptions)


def create_tool_metadata(
    name: str,
    description: str,
    params: Dict[str, Any],
    returns: str = "Any",
    examples: Optional[List[str]] = None,
    tags: Optional[List[str]] = None
) -> Dict:
    """Create standardized tool metadata."""
    return {
        'name': name,
        'description': description,
        'params': params,
        'returns': returns,
        'examples': examples or [],
        'tags': tags or [],
        'created_at': format_timestamp(),
        'version': '1.0.0'
    }


class ToolValidator:
    """Validate tool inputs and outputs."""
    
    @staticmethod
    def validate_type(value: Any, expected_type: type) -> bool:
        """Validate that a value matches the expected type."""
        return isinstance(value, expected_type)
    
    @staticmethod
    def validate_schema(data: Dict, schema: Dict) -> bool:
        """Validate data against a schema."""
        for key, expected_type in schema.items():
            if key not in data:
                return False
            if not ToolValidator.validate_type(data[key], expected_type):
                return False
        return True
    
    @staticmethod
    def validate_range(value: Union[int, float], min_val: Optional[Union[int, float]] = None, max_val: Optional[Union[int, float]] = None) -> bool:
        """Validate that a value is within a range."""
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True


# Export all utilities
__all__ = [
    'MCPError',
    'ToolError',
    'ValidationError',
    'retry',
    'cache_result',
    'validate_params',
    'measure_time',
    'error_handler',
    'timer',
    'MessageQueue',
    'Registry',
    'StateManager',
    'parse_json_safe',
    'format_timestamp',
    'truncate_text',
    'merge_dicts',
    'batch_process',
    'run_parallel',
    'create_tool_metadata',
    'ToolValidator'
]


if __name__ == "__main__":
    print("MCP Utils Module - Ready")
    print("Available utilities:")
    for item in __all__:
        print(f"  - {item}")
