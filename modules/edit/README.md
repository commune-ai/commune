# Edit Module
The Edit module provides a powerful interface for programmatically editing files using language models.
## Features
- File manipulation tools (add, delete, modify lines)
- LLM-powered code editing
- Interactive confirmation of changes
- Automatic file backups
- Support for multiple language models
## Usage
```python
import commune as c
# Initialize the Edit module
edit = c.module('edit')
# Make changes to a file
result = edit.forward(
    text="your editing instruction",
    path="path/to/your/file",
    model="anthropic/claude-3.5-sonnet"
)
```
## Available Tools
- `add_lines`: Add lines at a specific position
- `add_lines_after`: Add content after specified text
- `add_lines_before`: Add content before specified text
- `add_lines_between`: Add content between two text markers
- `add_file`: Create a new file with content
- `delete_lines`: Remove lines from specified positions
- `delete_file`: Delete a file or directory
- `delete_between`: Remove content between two text markers
## Safety Features
- Creates automatic backups before modifications
- Interactive confirmation for each operation
- Validation of file changes