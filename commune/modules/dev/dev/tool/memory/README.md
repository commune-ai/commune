
# Memory Tool

A flexible memory management system for AI applications that provides both short-term and long-term memory capabilities.

## Features

- **Short-term Memory**: In-memory storage with automatic expiration
- **Long-term Memory**: Persistent file-based storage
- **Relevance Filtering**: Find memories most relevant to a query
- **Memory Management**: Automatic cleanup of expired items
- **Memory Search**: Search through stored memories
- **Memory Summarization**: Generate summaries of stored memories

## Usage

```python
import commune as c

# Initialize the memory tool
memory = c.module('dev.tool.memory')()

# Store information in short-term memory (expires after default TTL)
memory.add_short_term("user_preference", {"theme": "dark", "language": "python"})

# Store information in long-term memory (persistent)
memory.add_long_term("project_requirements", {
    "name": "AI Assistant",
    "features": ["code generation", "memory management", "file editing"]
})

# Retrieve memories
user_pref = memory.get_short_term("user_preference")
requirements = memory.get_long_term("project_requirements")

# Filter a list of items by relevance to a query
files = ["main.py", "utils.py", "memory.py", "database.py"]
relevant_files = memory.forward(files, query="memory management", n=2)
# Returns: ["memory.py", "utils.py"]

# Search long-term memory
relevant_memories = memory.search_long_term("project features")

# Generate a summary of memories
summary = memory.summarize_memories(query="user preferences")
```

## Integration with Agent Module

The Memory tool is designed to work seamlessly with the Agent module:

```python
dev = c.module('dev')()
dev.set_memory(c.module('dev.tool.memory')())

# Now Agent will use the memory tool to maintain context
```

## Advanced Features

### Memory Eviction Policies

When short-term memory reaches capacity, items are evicted using a Least Recently Used (LRU) strategy.

### Memory Persistence

Long-term memories are stored as JSON files in the specified directory (default: `~/.commune/memory/long_term`).

### Relevance Scoring

The tool uses LLM-based relevance scoring to find the most relevant memories for a given query.
