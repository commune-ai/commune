# Agent Module

The Agent module is an advanced code generation and editing toolkit powered by LLMs. It provides a powerful interface for generating, editing, and managing code through natural language prompts.

## Features

- **Code Generation**: Create code from scratch based on natural language descriptions
- **Code Editing**: Modify existing codebases with natural language instructions
- **File Selection**: Find relevant files based on semantic understanding
- **Function Calling**: Dynamically incorporate function results into prompts
- **Interactive Workflow**: Preview and confirm changes before applying them
- **Memory Management**: Maintain context across interactions using short-term and long-term memory

## Quick Start

```python
import commune as c

# Initialize the dev module
dev = c.module('dev')()

# Generate code from a prompt
dev.forward("Create a Python function that calculates Fibonacci numbers")

# Edit existing code
dev.forward("Add error handling to the function", to="./path/to/file.py")

# Use the toolbox for more guidance
toolbox = c.module('dev.tool.toolbox')()
toolbox.help()
```

## Core Components

1. **Agent**: Main module for code generation and editing
2. **Edit**: Specialized module for editing existing code
3. **Select**: Tool for finding relevant files based on queries
4. **Toolbox**: Guide and helper for using all dev tools effectively
5. **Memory**: Tool for maintaining context across interactions

## Usage Examples

### Generate a New Module

```python
dev.forward(
    "Create a REST API with endpoints for user management (create, read, update, delete)",
    to="./api"
)
```

### Edit Existing Code

```python
dev.forward(
    "Add input validation and error handling to this function",
    to="./utils/helpers.py"
)
```

### Find Relevant Files

```python
select = c.module('dev.tool.select')()
files = c.files("./project")
auth_files = select.forward(
    options=files,
    query="files related to authentication"
)
```

### Function Calling

```python
dev.forward("Document these functions: @/get_text ./utils/helpers.py")
```

### Using Memory

```python
# Initialize the memory tool
memory = c.module('dev.tool.memory')()

# Store information in short-term memory
memory.add_short_term("user_preference", {"theme": "dark", "language": "python"})

# Store information in long-term memory
memory.add_long_term("project_requirements", {
    "name": "AI Assistant",
    "features": ["code generation", "memory management", "file editing"]
})

# Integrate memory with Agent module
dev = c.module('dev')()
dev.set_memory(memory)
```

## Advanced Configuration

The Agent module can be configured with various parameters:

- `model`: Choose which LLM to use (default: "anthropic/claude-3.7-sonnet")
- `temperature`: Control creativity (0.0-1.0)
- `max_tokens`: Limit the response size
- `verbose`: Enable/disable detailed output

## Available Tools

The Agent module includes several specialized tools:

- **cmd**: Execute shell commands
- **create_file**: Create new files with specified content
- **delete_file/delete_folder**: Remove files or directories
- **insert_text**: Insert content between anchors in a file
- **select_files**: Find relevant files based on queries
- **summarize/summarize_file**: Generate summaries of content
- **web_scraper**: Search the web for information

## Getting Help

For more detailed guidance, use the Toolbox:

```python
toolbox = c.module('dev.tool.toolbox')()

# General help
toolbox.help()

# Tool-specific examples
toolbox.example("dev")
toolbox.example("edit")
toolbox.example("select")

# Detailed guides
toolbox.dev_guide()
toolbox.edit_guide()
toolbox.select_guide()

# Quick start guide
toolbox.quick_start()

# Function calling guide
toolbox.function_calling_guide()
```

## Integration with Other Modules

The Agent module is designed to work seamlessly with other modules in the ecosystem:

```python
# Web scraping integration
web_scraper = c.module('web_scraper')()
results = web_scraper.forward("latest AI developments")
dev.forward(f"Create a summary of these developments: {results['context']}")

# Memory integration
memory = c.module('dev.tool.memory')()
dev.set_memory(memory)
```

## Contributing

Contributions to the Agent module are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
