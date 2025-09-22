#!/usr/bin/env python3
"""
MCP (Module Control Protocol) - Base module for commune-style architecture

A Leonardo da Vinci inspired approach to modular Python development.
Simple, elegant, and powerful.
"""

import os
import sys
import json
import importlib
import inspect
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class MCP:
    """
    Base class for all modules in the commune architecture.
    Provides core functionality for module management, communication, and execution.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the MCP module with optional configuration."""
        self.config = config or {}
        self.name = self.__class__.__name__.lower()
        self.path = Path(__file__).parent
        self._modules = {}
        self._tools = {}
        
    @classmethod
    def module_info(cls) -> Dict[str, Any]:
        """Get information about this module."""
        return {
            'name': cls.__name__,
            'doc': cls.__doc__,
            'methods': cls.get_methods(),
            'path': str(Path(inspect.getfile(cls)))
        }
    
    @classmethod
    def get_methods(cls) -> List[str]:
        """Get all public methods of this module."""
        return [m for m in dir(cls) if not m.startswith('_') and callable(getattr(cls, m))]
    
    def call(self, method: str, *args, **kwargs) -> Any:
        """Call a method on this module."""
        if hasattr(self, method):
            return getattr(self, method)(*args, **kwargs)
        raise AttributeError(f"Module {self.name} has no method {method}")
    
    def register_tool(self, name: str, func: callable, metadata: Optional[Dict] = None):
        """Register a tool/function that can be called."""
        self._tools[name] = {
            'func': func,
            'metadata': metadata or {},
            'signature': str(inspect.signature(func))
        }
    
    def get_tool(self, name: str) -> Optional[Dict]:
        """Get a registered tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self._tools.keys())
    
    def load_module(self, module_path: str) -> Any:
        """Dynamically load a module."""
        try:
            if module_path in self._modules:
                return self._modules[module_path]
            
            spec = importlib.util.spec_from_file_location("module", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            self._modules[module_path] = module
            return module
        except Exception as e:
            print(f"Error loading module {module_path}: {e}")
            return None
    
    def save_config(self, path: Optional[str] = None):
        """Save configuration to a JSON file."""
        path = path or f"{self.name}_config.json"
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_config(self, path: Optional[str] = None):
        """Load configuration from a JSON file."""
        path = path or f"{self.name}_config.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.config = json.load(f)
        return self.config
    
    def __repr__(self) -> str:
        return f"<MCP Module: {self.name}>"
    
    def __str__(self) -> str:
        return self.name


class Agent(MCP):
    """
    Agent class that extends MCP for autonomous operations.
    Inspired by Mr. Robot's approach to problem-solving.
    """
    
    def __init__(self, goal: str = None, **kwargs):
        super().__init__(**kwargs)
        self.goal = goal
        self.history = []
        self.state = {}
    
    def think(self, context: Any) -> Dict:
        """Process context and generate thoughts."""
        thought = {
            'context': str(context),
            'goal': self.goal,
            'timestamp': self._timestamp()
        }
        self.history.append(thought)
        return thought
    
    def act(self, action: str, params: Dict = None) -> Any:
        """Execute an action with given parameters."""
        params = params or {}
        result = {
            'action': action,
            'params': params,
            'timestamp': self._timestamp()
        }
        
        if action in self._tools:
            tool = self._tools[action]
            result['output'] = tool['func'](**params)
        else:
            result['output'] = self.call(action, **params)
        
        self.history.append(result)
        return result
    
    def _timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


class Toolbelt(MCP):
    """
    Toolbelt class for managing and organizing tools.
    A Leonardo-style workshop of digital instruments.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.categories = {}
    
    def add_category(self, name: str, description: str = ""):
        """Add a new tool category."""
        self.categories[name] = {
            'description': description,
            'tools': []
        }
    
    def add_tool_to_category(self, category: str, tool_name: str):
        """Add a tool to a specific category."""
        if category in self.categories:
            if tool_name in self._tools:
                self.categories[category]['tools'].append(tool_name)
    
    def get_category_tools(self, category: str) -> List[str]:
        """Get all tools in a category."""
        if category in self.categories:
            return self.categories[category]['tools']
        return []
    
    def search_tools(self, query: str) -> List[str]:
        """Search for tools matching a query."""
        matches = []
        query_lower = query.lower()
        
        for name, tool in self._tools.items():
            if query_lower in name.lower():
                matches.append(name)
            elif 'metadata' in tool and 'description' in tool['metadata']:
                if query_lower in tool['metadata']['description'].lower():
                    matches.append(name)
        
        return matches


def create_module(name: str, base_class: type = MCP, methods: Dict = None) -> type:
    """
    Factory function to create new module classes dynamically.
    
    Args:
        name: Name of the new module class
        base_class: Base class to inherit from (default: MCP)
        methods: Dictionary of method names to functions
    
    Returns:
        New module class
    """
    methods = methods or {}
    return type(name, (base_class,), methods)


def commune():
    """
    Main entry point for the commune system.
    Simple, elegant, powerful.
    """
    print("""
    ╔═══════════════════════════════════════╗
    ║         COMMUNE MCP SYSTEM            ║
    ║    'Simplicity is the ultimate        ║
    ║         sophistication'                ║
    ║           - Leonardo da Vinci          ║
    ╚═══════════════════════════════════════╝
    """)
    
    # Initialize base system
    base = MCP()
    print(f"\n✓ MCP Base initialized: {base}")
    
    # Create an agent
    agent = Agent(goal="Build amazing things")
    print(f"✓ Agent created with goal: {agent.goal}")
    
    # Create a toolbelt
    tools = Toolbelt()
    print(f"✓ Toolbelt ready")
    
    print("\n🚀 System ready. Let's build something beautiful.\n")
    
    return {
        'base': base,
        'agent': agent,
        'tools': tools
    }


if __name__ == "__main__":
    # Run the commune system
    system = commune()
    
    # Example usage
    print("Example Usage:")
    print("-" * 40)
    
    # Get module info
    info = MCP.module_info()
    print(f"Module: {info['name']}")
    print(f"Methods: {', '.join(info['methods'][:5])}...")
    
    # Create a custom module
    CustomModule = create_module(
        'CustomModule',
        MCP,
        {'custom_method': lambda self: "Hello from custom module!"}
    )
    
    custom = CustomModule()
    print(f"\nCustom module created: {custom}")
    
    print("\n✨ MCP system is running. Build something amazing! ✨")
