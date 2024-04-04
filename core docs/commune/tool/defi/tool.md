# Tool Module

This is a Python module that defines a class called `Tool` which is meant to be used as a base for any tool that performs some sort of calculation or function, hence a toolkit. 

## Features

- Perform calculations using the `call()` function.
- Fetch list of all tools available.
- Fetch information (metadata) about each tool.
- Fetch filepath and code of a specific tool.
- Fetch general schema of a specific tool.

## Usage

Here is a quick example of how to use the code:

```python
# Create object of the Tool class
tool = Tool()

# Call method
res = tool.call(x=2, y=3) 
print(res) # Expected output: 7 (2*2 + 3)

# Get list of all tools
tool_list = Tool.tool_list() 
print(tool_list)

# Get information about a specific tool
info = Tool.info()
print(info)

# Get file path of a specific tool
filepath = Tool.filepath()
print(filepath)

# Get code of a specific tool
code = Tool.code()
print(code)

# Get general schema of a specific tool
schema = Tool.get_general_schema()
print(schema)
```
