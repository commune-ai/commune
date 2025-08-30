
import commune as c
import os
import json
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

class Memory:
    """
    A memory management tool that provides both short-term and long-term memory capabilities.
    
    This tool helps maintain context across interactions by:
    - Storing temporary information in short-term memory (in-memory)
    - Persisting important information in long-term memory (file-based)
    - Retrieving and filtering memories based on relevance
    - Managing memory expiration and prioritization
    """
    
    
    def __init__(
        self,
        path: str = "~/.commune/agent/memory",
        short_term_chars: int = 100000,
        model: str = 'model.openrouter',
        item_size = 1000,  # Size of each memory item in characters
        **kwargs
    ):
        """
        Initialize the Memory module.
        
        Args:
            path: Path to store memories
            short_term_chars: Maximum characters in short-term memory
            model: Model to use for relevance scoring
            item_size: Size of each memory item in characters
            **kwargs: Additional arguments to pass to the model
        """
        self.store = c.mod('store')(path)
        self.short_term_chars = short_term_chars
        self.item_size = item_size
        self.model = c.mod(model)(**kwargs)
        self.short_term_memory = {}  # In-memory storage
        self.memory_timestamps = {}  # Track when memories were added
        
    def add_memory(self, key: str, content: Any, memory_type: str = 'short') -> Dict[str, Any]:
        """
        Add a memory item.
        
        Args:
            key: Unique identifier for the memory
            content: Content to store
            memory_type: 'short' or 'long' term memory
        """
        timestamp = time.time()
        memory_item = {
            'key': key,
            'content': content,
            'timestamp': timestamp,
            'type': memory_type,
            'access_count': 0
        }
        
        if memory_type == 'short':
            self.short_term_memory[key] = memory_item
            self.memory_timestamps[key] = timestamp
            self._manage_short_term_capacity()
        else:
            # Store in long-term memory
            self.store.put(f'long_term/{key}', memory_item)
            
        return {'success': True, 'key': key, 'type': memory_type}
    
    def _manage_short_term_capacity(self):
        """
        Manage short-term memory capacity using LRU eviction.
        """
        total_chars = sum(len(str(m['content'])) for m in self.short_term_memory.values())
        
        if total_chars > self.short_term_chars:
            # Sort by timestamp (oldest first)
            sorted_keys = sorted(self.memory_timestamps.keys(), 
                               key=lambda k: self.memory_timestamps[k])
            
            while total_chars > self.short_term_chars and sorted_keys:
                key_to_remove = sorted_keys.pop(0)
                if key_to_remove in self.short_term_memory:
                    del self.short_term_memory[key_to_remove]
                    del self.memory_timestamps[key_to_remove]
                    total_chars = sum(len(str(m['content'])) 
                                    for m in self.short_term_memory.values())
    
    def get_memory(self, key: str, memory_type: str = 'short') -> Optional[Any]:
        """
        Retrieve a specific memory by key.
        """
        if memory_type == 'short':
            memory = self.short_term_memory.get(key)
            if memory:
                memory['access_count'] += 1
                return memory['content']
        else:
            memory = self.store.get(f'long_term/{key}')
            if memory:
                memory['access_count'] += 1
                self.store.put(f'long_term/{key}', memory)
                return memory['content']
        return None
    
    def search_memories(self, query: str, n: int = 5, memory_type: str = 'all') -> List[Dict[str, Any]]:
        """
        Search memories using hierarchical search with the model.
        
        Args:
            query: Search query
            n: Number of results to return
            memory_type: 'short', 'long', or 'all'
        """
        memories = []
        
        # Collect memories based on type
        if memory_type in ['short', 'all']:
            for key, memory in self.short_term_memory.items():
                memories.append({
                    'key': key,
                    'content': memory['content'],
                    'type': 'short',
                    'timestamp': memory['timestamp']
                })
        
        if memory_type in ['long', 'all']:
            long_term_keys = self.store.ls('long_term/')
            for key in long_term_keys:
                memory = self.store.get(key)
                if memory:
                    memories.append({
                        'key': key.replace('long_term/', ''),
                        'content': memory['content'],
                        'type': 'long',
                        'timestamp': memory['timestamp']
                    })
        
        if not memories:
            return []
        
        # Use model for hierarchical relevance scoring
        prompt = f"""
        Given the search query, rank these memories by relevance.
        Return the indices of the {n} most relevant memories in order of relevance.
        
        Query: {query}
        
        Memories:
        """
        
        for i, mem in enumerate(memories):
            prompt += f"\n{i}. Key: {mem['key']}, Content: {str(mem['content'])[:200]}..."
        
        prompt += f"\n\nReturn ONLY a JSON list of the {n} most relevant memory indices: [index1, index2, ...]\n"
        
        try:
            response = self.model.forward(prompt, temperature=0.3)
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*?\]', response)
            if json_match:
                indices = json.loads(json_match.group())
                # Return the selected memories
                return [memories[i] for i in indices if i < len(memories)][:n]
        except Exception as e:
            c.print(f"Error in hierarchical search: {e}", color='yellow')
            # Fallback to simple sorting by timestamp
            memories.sort(key=lambda x: x['timestamp'], reverse=True)
            return memories[:n]
    
    def summarize_memories(self, query: Optional[str] = None, memory_type: str = 'all') -> str:
        """
        Generate a summary of memories, optionally filtered by query.
        """
        if query:
            memories = self.search_memories(query, n=10, memory_type=memory_type)
        else:
            memories = self.search_memories('', n=20, memory_type=memory_type)
        
        if not memories:
            return "No memories found."
        
        prompt = "Summarize these memories into a coherent overview:\n\n"
        for mem in memories:
            prompt += f"- {mem['key']}: {str(mem['content'])[:200]}...\n"
        
        prompt += "\nProvide a concise summary:"
        
        summary = self.model.forward(prompt, temperature=0.5)
        return summary
    
    def clear_short_term(self) -> Dict[str, Any]:
        """
        Clear all short-term memories.
        """
        count = len(self.short_term_memory)
        self.short_term_memory.clear()
        self.memory_timestamps.clear()
        return {'success': True, 'cleared': count}
    
    def migrate_to_long_term(self, key: str) -> Dict[str, Any]:
        """
        Move a memory from short-term to long-term storage.
        """
        if key in self.short_term_memory:
            memory = self.short_term_memory[key]
            self.store.put(f'long_term/{key}', memory)
            del self.short_term_memory[key]
            del self.memory_timestamps[key]
            return {'success': True, 'key': key, 'migrated': True}
        return {'success': False, 'error': 'Memory not found in short-term storage'}
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about memory usage.
        """
        short_term_count = len(self.short_term_memory)
        short_term_chars = sum(len(str(m['content'])) for m in self.short_term_memory.values())
        long_term_count = len(self.store.ls('long_term/'))
        
        return {
            'short_term': {
                'count': short_term_count,
                'characters': short_term_chars,
                'capacity': self.short_term_chars,
                'usage_percent': (short_term_chars / self.short_term_chars) * 100
            },
            'long_term': {
                'count': long_term_count
            }
        }
    
    def forward(self, action: str = 'search', **kwargs) -> Any:
        """
        Main interface for the memory module.
        
        Actions:
        - add: Add a memory (requires: key, content, memory_type)
        - get: Get a specific memory (requires: key, memory_type)
        - search: Search memories (requires: query, n, memory_type)
        - summarize: Summarize memories (optional: query, memory_type)
        - stats: Get memory statistics
        - clear: Clear short-term memory
        - migrate: Migrate memory to long-term (requires: key)
        """
        if action == 'add':
            return self.add_memory(**kwargs)
        elif action == 'get':
            return self.get_memory(**kwargs)
        elif action == 'search':
            return self.search_memories(**kwargs)
        elif action == 'summarize':
            return self.summarize_memories(**kwargs)
        elif action == 'stats':
            return self.get_memory_stats()
        elif action == 'clear':
            return self.clear_short_term()
        elif action == 'migrate':
            return self.migrate_to_long_term(**kwargs)
        else:
            return {'error': f'Unknown action: {action}'}
