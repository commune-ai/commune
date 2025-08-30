# Poetry Module

A comprehensive Poetry module for Commune that provides all Poetry functionality in a single Python class.

## Features

- **Project Management**: Create, initialize, and manage Poetry projects
- **Dependency Management**: Add, remove, update, and lock dependencies
- **Environment Management**: Manage virtual environments and Python versions
- **Building & Publishing**: Build and publish packages to PyPI or custom repositories
- **Configuration**: Manage Poetry and project configuration
- **Utilities**: Various helper methods for common tasks

## Installation

The module requires Poetry to be installed. If it's not installed, you can use the built-in installer:

```python
import commune as c

poetry = c.module('poetry')()
result = poetry.install_poetry()
print(result)
```

## Usage

### Basic Usage

```python
import commune as c

# Initialize Poetry module for current directory
poetry = c.module('poetry')()

# Or specify a project path
poetry = c.module('poetry')(project_path='~/my-project')
```

### Creating a New Project

```python
# Create a new Poetry project
result = poetry.new('my-awesome-project', src=True)

# Or create a basic project with structure
result = poetry.create_basic_project(
    name='my-project',
    description='An awesome Python project',
    author='Your Name <you@example.com>',
    dependencies={'requests': '^2.28.0'},
    dev_dependencies={'pytest': '^7.0.0'}
)
```

### Managing Dependencies

```python
# Add dependencies
poetry.add('requests')
poetry.add(['numpy', 'pandas'], group='data')
poetry.add('pytest', dev=True)

# Remove dependencies
poetry.remove('requests')

# Update dependencies
poetry.update()  # Update all
poetry.update('requests')  # Update specific package

# Show dependency information
poetry.show(tree=True)  # Show dependency tree
poetry.show('requests')  # Show specific package info
```

### Environment Management

```python
# Get environment info
info = poetry.env_info()

# List all environments
envs = poetry.env_list()

# Use specific Python version
poetry.env_use('python3.9')

# Remove environments
poetry.env_remove('python3.8')
poetry.env_remove(all=True)  # Remove all
```

### Building and Publishing

```python
# Build the package
poetry.build(format='wheel')

# Publish to PyPI
poetry.publish(
    username='your-username',
    password='your-password'
)

# Publish to custom repository
poetry.publish(
    repository='my-repo',
    username='user',
    password='pass'
)
```

### Running Commands

```python
# Run commands in Poetry environment
poetry.run('python script.py')
poetry.run('pytest tests/')

# Spawn a shell in the environment
poetry.shell()
```

### Configuration

```python
# List configuration
config = poetry.config(list_=True)

# Set configuration
poetry.config('virtualenvs.in-project', 'true')

# Get specific config value
result = poetry.config('virtualenvs.path')
```

### Source Management

```python
# Add a custom package source
poetry.source_add(
    name='private',
    url='https://my-private-repo.com/simple/',
    secondary=True
)

# Show sources
poetry.source_show()

# Remove a source
poetry.source_remove('private')
```

### Utility Methods

```python
# Check if current directory is a Poetry project
is_poetry = poetry.is_poetry_project()

# Get project information
project_info = poetry.get_project_info()
print(f"Project: {project_info['name']} v{project_info['version']}")

# Get current dependencies
deps = poetry.get_dependencies()
dev_deps = poetry.get_dependencies(dev=True)

# Get virtualenv path
venv_path = poetry.get_virtualenv_path()

# Get Python version
python_version = poetry.get_python_version()
```

### Export Dependencies

```python
# Export to requirements.txt
poetry.export(
    format='requirements.txt',
    output='requirements.txt',
    without_hashes=True
)

# Export only production dependencies
poetry.export(
    output='requirements-prod.txt',
    without=['dev', 'test']
)
```

### Version Management

```python
# Show current version
version = poetry.version()

# Update version
poetry.version('1.2.0')  # Set specific version
poetry.version('patch')  # Bump patch version
poetry.version('minor')  # Bump minor version
poetry.version('major')  # Bump major version
```

### Cache Management

```python
# List caches
caches = poetry.cache_list()

# Clear specific cache
poetry.cache_clear('pypi')

# Clear all caches
poetry.cache_clear('pypi', all=True)
```

## Advanced Usage

### Using the Forward Method

The `forward` method provides a unified interface for all Poetry commands:

```python
# Execute any Poetry command
result = poetry.forward('install', no_dev=True)
result = poetry.forward('add', packages=['requests', 'numpy'])
result = poetry.forward('show', tree=True)
```

### Direct Command Execution

For commands not covered by specific methods, you can use the internal command runner:

```python
# Run any Poetry command directly
result = poetry._run_poetry_command(['config', '--list'])
result = poetry._run_poetry_command('check')
```

### Error Handling

All methods return a dictionary with execution results:

```python
result = poetry.add('invalid-package-name-12345')

if result['success']:
    print("Package added successfully")
else:
    print(f"Error: {result.get('error', 'Unknown error')}")
    print(f"stderr: {result.get('stderr', '')}")
```

## Examples

### Complete Project Setup

```python
import commune as c

# Initialize Poetry module
poetry = c.module('poetry')()

# Create a new project
project_name = 'my-ml-project'
poetry.create_basic_project(
    name=project_name,
    description='A machine learning project',
    author='ML Developer <ml@example.com>',
    python_version='^3.8',
    dependencies={
        'numpy': '^1.24.0',
        'pandas': '^2.0.0',
        'scikit-learn': '^1.3.0'
    },
    dev_dependencies={
        'pytest': '^7.4.0',
        'black': '^23.0.0',
        'flake8': '^6.0.0'
    }
)

# Change to project directory
import os
os.chdir(project_name)

# Install dependencies
poetry.install()

# Add additional dependencies
poetry.add('matplotlib', group='visualization')
poetry.add('jupyter', dev=True)

# Check project
poetry.check()

# Show project info
info = poetry.get_project_info()
print(f"Created project: {info['name']} v{info['version']}")
```

### CI/CD Integration

```python
# For CI/CD pipelines
def setup_ci_environment():
    poetry = c.module('poetry')()
    
    # Install dependencies without dev packages
    result = poetry.install(no_dev=True, no_root=True)
    if not result['success']:
        raise Exception(f"Failed to install dependencies: {result['error']}")
    
    # Export requirements for Docker
    poetry.export(
        output='requirements.txt',
        without_hashes=True,
        without=['dev', 'test']
    )
    
    # Run tests
    test_result = poetry.run('pytest tests/ -v')
    
    # Build package
    build_result = poetry.build()
    
    return {
        'install': result,
        'tests': test_result,
        'build': build_result
    }
```

## Notes

- The module assumes Poetry is installed or will help install it
- All paths are relative to the `project_path` specified during initialization
- Methods return dictionaries with execution results for easy error handling
- The module provides both high-level methods and low-level command execution

## Contributing

Feel free to extend this module with additional Poetry functionality or utility methods as needed.
