import os
import subprocess
import json
import toml
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import commune as c


class Poetry(c.Module):
    """
    A comprehensive Poetry module that provides all Poetry functionality in a single class.
    This module wraps Poetry CLI commands and provides additional utilities for Python project management.
    """
    
    def __init__(self, project_path: str = './', **kwargs):
        """
        Initialize the Poetry module.
        
        Args:
            project_path: Path to the project directory (default: current directory)
        """
        super().__init__(**kwargs)
        self.project_path = os.path.abspath(os.path.expanduser(project_path))
        self.pyproject_path = os.path.join(self.project_path, 'pyproject.toml')
        
    def _run_poetry_command(self, command: Union[str, List[str]], 
                          capture_output: bool = True,
                          check: bool = True,
                          **kwargs) -> Dict[str, Any]:
        """
        Execute a poetry command and return the result.
        
        Args:
            command: Poetry command to execute (string or list)
            capture_output: Whether to capture stdout/stderr
            check: Whether to raise exception on non-zero exit code
            **kwargs: Additional arguments for subprocess.run
            
        Returns:
            Dictionary with command execution results
        """
        if isinstance(command, str):
            command = command.split()
        
        # Ensure 'poetry' is the first element
        if command[0] != 'poetry':
            command = ['poetry'] + command
            
        try:
            result = subprocess.run(
                command,
                cwd=self.project_path,
                capture_output=capture_output,
                text=True,
                check=check,
                **kwargs
            )
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout if capture_output else None,
                'stderr': result.stderr if capture_output else None,
                'returncode': result.returncode,
                'command': ' '.join(command)
            }
            
        except subprocess.CalledProcessError as e:
            return {
                'success': False,
                'stdout': e.stdout if hasattr(e, 'stdout') else None,
                'stderr': e.stderr if hasattr(e, 'stderr') else None,
                'returncode': e.returncode,
                'command': ' '.join(command),
                'error': str(e)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'command': ' '.join(command)
            }
    
    def new(self, name: str, path: Optional[str] = None, 
            src: bool = False, readme: bool = True) -> Dict[str, Any]:
        """
        Create a new Python package.
        
        Args:
            name: Package name
            path: Path where to create the package
            src: Use src layout
            readme: Create README file
            
        Returns:
            Command execution result
        """
        cmd = ['new', name]
        if path:
            cmd.extend(['--path', path])
        if src:
            cmd.append('--src')
        if not readme:
            cmd.append('--no-readme')
            
        return self._run_poetry_command(cmd)
    
    def init(self, name: Optional[str] = None, 
             description: Optional[str] = None,
             author: Optional[str] = None,
             python: Optional[str] = None,
             dependency: Optional[List[str]] = None,
             dev_dependency: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Initialize a new pyproject.toml file.
        
        Args:
            name: Package name
            description: Package description
            author: Author name
            python: Python version requirement
            dependency: List of dependencies
            dev_dependency: List of dev dependencies
            
        Returns:
            Command execution result
        """
        cmd = ['init']
        if name:
            cmd.extend(['--name', name])
        if description:
            cmd.extend(['--description', description])
        if author:
            cmd.extend(['--author', author])
        if python:
            cmd.extend(['--python', python])
        if dependency:
            for dep in dependency:
                cmd.extend(['--dependency', dep])
        if dev_dependency:
            for dep in dev_dependency:
                cmd.extend(['--dev-dependency', dep])
                
        return self._run_poetry_command(cmd)
    
    def install(self, no_dev: bool = False, no_root: bool = False,
                extras: Optional[List[str]] = None,
                with_: Optional[List[str]] = None,
                without: Optional[List[str]] = None,
                only: Optional[List[str]] = None,
                sync: bool = False) -> Dict[str, Any]:
        """
        Install project dependencies.
        
        Args:
            no_dev: Do not install dev dependencies (deprecated)
            no_root: Do not install the project package
            extras: Extras to install
            with_: Include optional dependency groups
            without: Exclude dependency groups
            only: Only install specified dependency groups
            sync: Synchronize the environment
            
        Returns:
            Command execution result
        """
        cmd = ['install']
        if no_dev:
            cmd.append('--no-dev')
        if no_root:
            cmd.append('--no-root')
        if sync:
            cmd.append('--sync')
        if extras:
            for extra in extras:
                cmd.extend(['--extras', extra])
        if with_:
            for group in with_:
                cmd.extend(['--with', group])
        if without:
            for group in without:
                cmd.extend(['--without', group])
        if only:
            for group in only:
                cmd.extend(['--only', group])
                
        return self._run_poetry_command(cmd)
    
    def add(self, packages: Union[str, List[str]], 
            group: Optional[str] = None,
            dev: bool = False,
            editable: bool = False,
            extras: Optional[List[str]] = None,
            optional: bool = False,
            python: Optional[str] = None,
            platform: Optional[str] = None,
            source: Optional[str] = None,
            allow_prereleases: bool = False,
            dry_run: bool = False,
            lock: bool = True) -> Dict[str, Any]:
        """
        Add dependencies to pyproject.toml.
        
        Args:
            packages: Package(s) to add
            group: Add to a specific dependency group
            dev: Add as development dependency
            editable: Add as editable dependency
            extras: Extras to enable for the dependency
            optional: Add as optional dependency
            python: Python version requirement
            platform: Platform requirement
            source: Name of the source to use
            allow_prereleases: Allow prereleases
            dry_run: Don't actually install, just show what would happen
            lock: Don't update the lock file
            
        Returns:
            Command execution result
        """
        if isinstance(packages, str):
            packages = [packages]
            
        cmd = ['add'] + packages
        
        if group:
            cmd.extend(['--group', group])
        if dev:
            cmd.append('--dev')
        if editable:
            cmd.append('--editable')
        if optional:
            cmd.append('--optional')
        if python:
            cmd.extend(['--python', python])
        if platform:
            cmd.extend(['--platform', platform])
        if source:
            cmd.extend(['--source', source])
        if allow_prereleases:
            cmd.append('--allow-prereleases')
        if dry_run:
            cmd.append('--dry-run')
        if not lock:
            cmd.append('--no-update')
        if extras:
            for extra in extras:
                cmd.extend(['--extras', extra])
                
        return self._run_poetry_command(cmd)
    
    def remove(self, packages: Union[str, List[str]],
               group: Optional[str] = None,
               dev: bool = False,
               dry_run: bool = False) -> Dict[str, Any]:
        """
        Remove dependencies from pyproject.toml.
        
        Args:
            packages: Package(s) to remove
            group: Remove from a specific dependency group
            dev: Remove from development dependencies
            dry_run: Don't actually remove, just show what would happen
            
        Returns:
            Command execution result
        """
        if isinstance(packages, str):
            packages = [packages]
            
        cmd = ['remove'] + packages
        
        if group:
            cmd.extend(['--group', group])
        if dev:
            cmd.append('--dev')
        if dry_run:
            cmd.append('--dry-run')
            
        return self._run_poetry_command(cmd)
    
    def update(self, packages: Optional[Union[str, List[str]]] = None,
               dry_run: bool = False,
               no_dev: bool = False,
               lock: bool = False,
               with_: Optional[List[str]] = None,
               without: Optional[List[str]] = None,
               only: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Update dependencies.
        
        Args:
            packages: Specific package(s) to update (None for all)
            dry_run: Don't actually update, just show what would happen
            no_dev: Do not update dev dependencies
            lock: Only update the lock file
            with_: Include optional dependency groups
            without: Exclude dependency groups
            only: Only update specified dependency groups
            
        Returns:
            Command execution result
        """
        cmd = ['update']
        
        if packages:
            if isinstance(packages, str):
                packages = [packages]
            cmd.extend(packages)
            
        if dry_run:
            cmd.append('--dry-run')
        if no_dev:
            cmd.append('--no-dev')
        if lock:
            cmd.append('--lock')
        if with_:
            for group in with_:
                cmd.extend(['--with', group])
        if without:
            for group in without:
                cmd.extend(['--without', group])
        if only:
            for group in only:
                cmd.extend(['--only', group])
                
        return self._run_poetry_command(cmd)
    
    def show(self, package: Optional[str] = None,
             no_dev: bool = False,
             tree: bool = False,
             latest: bool = False,
             outdated: bool = False,
             all: bool = False,
             top_level: bool = False,
             with_: Optional[List[str]] = None,
             without: Optional[List[str]] = None,
             only: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Show information about packages.
        
        Args:
            package: Specific package to show info for
            no_dev: Do not show dev dependencies
            tree: Show dependency tree
            latest: Show latest version
            outdated: Show only outdated packages
            all: Show all packages (including transitive)
            top_level: Only show top-level dependencies
            with_: Include optional dependency groups
            without: Exclude dependency groups
            only: Only show specified dependency groups
            
        Returns:
            Command execution result
        """
        cmd = ['show']
        
        if package:
            cmd.append(package)
        if no_dev:
            cmd.append('--no-dev')
        if tree:
            cmd.append('--tree')
        if latest:
            cmd.append('--latest')
        if outdated:
            cmd.append('--outdated')
        if all:
            cmd.append('--all')
        if top_level:
            cmd.append('--top-level')
        if with_:
            for group in with_:
                cmd.extend(['--with', group])
        if without:
            for group in without:
                cmd.extend(['--without', group])
        if only:
            for group in only:
                cmd.extend(['--only', group])
                
        return self._run_poetry_command(cmd)
    
    def build(self, format: Optional[str] = None,
              output: Optional[str] = None) -> Dict[str, Any]:
        """
        Build the package.
        
        Args:
            format: Build format (wheel or sdist)
            output: Output directory
            
        Returns:
            Command execution result
        """
        cmd = ['build']
        
        if format:
            cmd.extend(['--format', format])
        if output:
            cmd.extend(['--output', output])
            
        return self._run_poetry_command(cmd)
    
    def publish(self, repository: Optional[str] = None,
                username: Optional[str] = None,
                password: Optional[str] = None,
                cert: Optional[str] = None,
                client_cert: Optional[str] = None,
                skip_existing: bool = False,
                dry_run: bool = False,
                build: bool = True) -> Dict[str, Any]:
        """
        Publish the package to a repository.
        
        Args:
            repository: Repository to publish to
            username: Username for authentication
            password: Password for authentication
            cert: Path to CA certificate
            client_cert: Path to client certificate
            skip_existing: Skip errors for existing packages
            dry_run: Don't actually publish
            build: Build the package before publishing
            
        Returns:
            Command execution result
        """
        cmd = ['publish']
        
        if repository:
            cmd.extend(['--repository', repository])
        if username:
            cmd.extend(['--username', username])
        if password:
            cmd.extend(['--password', password])
        if cert:
            cmd.extend(['--cert', cert])
        if client_cert:
            cmd.extend(['--client-cert', client_cert])
        if skip_existing:
            cmd.append('--skip-existing')
        if dry_run:
            cmd.append('--dry-run')
        if not build:
            cmd.append('--no-build')
            
        return self._run_poetry_command(cmd)
    
    def config(self, key: Optional[str] = None,
               value: Optional[str] = None,
               list_: bool = False,
               unset: bool = False,
               local: bool = False) -> Dict[str, Any]:
        """
        Manage Poetry configuration.
        
        Args:
            key: Configuration key
            value: Configuration value to set
            list_: List configuration settings
            unset: Unset configuration key
            local: Use local configuration
            
        Returns:
            Command execution result
        """
        cmd = ['config']
        
        if list_:
            cmd.append('--list')
        elif unset and key:
            cmd.extend(['--unset', key])
        elif key:
            cmd.append(key)
            if value:
                cmd.append(value)
                
        if local:
            cmd.append('--local')
            
        return self._run_poetry_command(cmd)
    
    def run(self, command: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Run a command in the Poetry environment.
        
        Args:
            command: Command to run
            
        Returns:
            Command execution result
        """
        if isinstance(command, list):
            command = ' '.join(command)
            
        return self._run_poetry_command(['run'] + command.split())
    
    def shell(self) -> Dict[str, Any]:
        """
        Spawn a shell within the Poetry environment.
        
        Returns:
            Command execution result
        """
        return self._run_poetry_command(['shell'], capture_output=False)
    
    def check(self) -> Dict[str, Any]:
        """
        Check the validity of pyproject.toml.
        
        Returns:
            Command execution result
        """
        return self._run_poetry_command(['check'])
    
    def search(self, query: str) -> Dict[str, Any]:
        """
        Search for packages on PyPI.
        
        Args:
            query: Search query
            
        Returns:
            Command execution result
        """
        return self._run_poetry_command(['search', query])
    
    def lock(self, no_update: bool = False,
             check: bool = False) -> Dict[str, Any]:
        """
        Lock dependencies.
        
        Args:
            no_update: Do not update locked versions
            check: Check if lock file is up to date
            
        Returns:
            Command execution result
        """
        cmd = ['lock']
        
        if no_update:
            cmd.append('--no-update')
        if check:
            cmd.append('--check')
            
        return self._run_poetry_command(cmd)
    
    def export(self, format: str = 'requirements.txt',
               output: Optional[str] = None,
               dev: bool = False,
               extras: Optional[List[str]] = None,
               without_hashes: bool = False,
               without_urls: bool = False,
               with_: Optional[List[str]] = None,
               without: Optional[List[str]] = None,
               only: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Export dependencies to other formats.
        
        Args:
            format: Export format (requirements.txt)
            output: Output file path
            dev: Include dev dependencies
            extras: Include extra dependencies
            without_hashes: Exclude hashes
            without_urls: Exclude URLs
            with_: Include optional dependency groups
            without: Exclude dependency groups
            only: Only export specified dependency groups
            
        Returns:
            Command execution result
        """
        cmd = ['export', '--format', format]
        
        if output:
            cmd.extend(['--output', output])
        if dev:
            cmd.append('--dev')
        if without_hashes:
            cmd.append('--without-hashes')
        if without_urls:
            cmd.append('--without-urls')
        if extras:
            for extra in extras:
                cmd.extend(['--extras', extra])
        if with_:
            for group in with_:
                cmd.extend(['--with', group])
        if without:
            for group in without:
                cmd.extend(['--without', group])
        if only:
            for group in only:
                cmd.extend(['--only', group])
                
        return self._run_poetry_command(cmd)
    
    def env_info(self) -> Dict[str, Any]:
        """
        Get environment information.
        
        Returns:
            Command execution result
        """
        return self._run_poetry_command(['env', 'info'])
    
    def env_list(self, full_path: bool = False) -> Dict[str, Any]:
        """
        List all virtualenvs.
        
        Args:
            full_path: Show full paths
            
        Returns:
            Command execution result
        """
        cmd = ['env', 'list']
        if full_path:
            cmd.append('--full-path')
            
        return self._run_poetry_command(cmd)
    
    def env_use(self, python: str) -> Dict[str, Any]:
        """
        Use a specific Python version.
        
        Args:
            python: Python executable or version
            
        Returns:
            Command execution result
        """
        return self._run_poetry_command(['env', 'use', python])
    
    def env_remove(self, python: Optional[str] = None,
                   all: bool = False) -> Dict[str, Any]:
        """
        Remove virtual environments.
        
        Args:
            python: Specific Python version to remove
            all: Remove all environments
            
        Returns:
            Command execution result
        """
        cmd = ['env', 'remove']
        
        if all:
            cmd.append('--all')
        elif python:
            cmd.append(python)
            
        return self._run_poetry_command(cmd)
    
    def cache_list(self) -> Dict[str, Any]:
        """
        List Poetry caches.
        
        Returns:
            Command execution result
        """
        return self._run_poetry_command(['cache', 'list'])
    
    def cache_clear(self, cache: str, all: bool = False) -> Dict[str, Any]:
        """
        Clear Poetry caches.
        
        Args:
            cache: Cache name to clear
            all: Clear all caches
            
        Returns:
            Command execution result
        """
        cmd = ['cache', 'clear', cache]
        if all:
            cmd.append('--all')
            
        return self._run_poetry_command(cmd)
    
    def version(self, version: Optional[str] = None,
                short: bool = False,
                dry_run: bool = False) -> Dict[str, Any]:
        """
        Show or update the version of the project.
        
        Args:
            version: New version to set
            short: Show only version number
            dry_run: Don't actually update version
            
        Returns:
            Command execution result
        """
        cmd = ['version']
        
        if version:
            cmd.append(version)
        if short:
            cmd.append('--short')
        if dry_run:
            cmd.append('--dry-run')
            
        return self._run_poetry_command(cmd)
    
    def about(self) -> Dict[str, Any]:
        """
        Show information about Poetry.
        
        Returns:
            Command execution result
        """
        return self._run_poetry_command(['about'])
    
    def self_update(self, version: Optional[str] = None,
                    preview: bool = False,
                    dry_run: bool = False) -> Dict[str, Any]:
        """
        Update Poetry itself.
        
        Args:
            version: Specific version to update to
            preview: Allow preview releases
            dry_run: Don't actually update
            
        Returns:
            Command execution result
        """
        cmd = ['self', 'update']
        
        if version:
            cmd.append(version)
        if preview:
            cmd.append('--preview')
        if dry_run:
            cmd.append('--dry-run')
            
        return self._run_poetry_command(cmd)
    
    def source_add(self, name: str, url: str,
                   default: bool = False,
                   secondary: bool = False) -> Dict[str, Any]:
        """
        Add a new repository source.
        
        Args:
            name: Source name
            url: Source URL
            default: Set as default source
            secondary: Set as secondary source
            
        Returns:
            Command execution result
        """
        cmd = ['source', 'add', name, url]
        
        if default:
            cmd.append('--default')
        if secondary:
            cmd.append('--secondary')
            
        return self._run_poetry_command(cmd)
    
    def source_show(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Show information about sources.
        
        Args:
            name: Specific source to show
            
        Returns:
            Command execution result
        """
        cmd = ['source', 'show']
        if name:
            cmd.append(name)
            
        return self._run_poetry_command(cmd)
    
    def source_remove(self, name: str) -> Dict[str, Any]:
        """
        Remove a repository source.
        
        Args:
            name: Source name to remove
            
        Returns:
            Command execution result
        """
        return self._run_poetry_command(['source', 'remove', name])
    
    # Utility methods
    
    def get_dependencies(self, dev: bool = False,
                        group: Optional[str] = None) -> Dict[str, str]:
        """
        Get current dependencies from pyproject.toml.
        
        Args:
            dev: Get dev dependencies
            group: Get dependencies from specific group
            
        Returns:
            Dictionary of package names to version specs
        """
        if not os.path.exists(self.pyproject_path):
            return {}
            
        with open(self.pyproject_path, 'r') as f:
            pyproject = toml.load(f)
            
        if group:
            deps = pyproject.get('tool', {}).get('poetry', {}).get('group', {}).get(group, {}).get('dependencies', {})
        elif dev:
            deps = pyproject.get('tool', {}).get('poetry', {}).get('dev-dependencies', {})
        else:
            deps = pyproject.get('tool', {}).get('poetry', {}).get('dependencies', {})
            
        # Remove python version as it's not a real dependency
        deps.pop('python', None)
        
        return deps
    
    def get_project_info(self) -> Dict[str, Any]:
        """
        Get project information from pyproject.toml.
        
        Returns:
            Dictionary with project metadata
        """
        if not os.path.exists(self.pyproject_path):
            return {}
            
        with open(self.pyproject_path, 'r') as f:
            pyproject = toml.load(f)
            
        poetry_section = pyproject.get('tool', {}).get('poetry', {})
        
        return {
            'name': poetry_section.get('name'),
            'version': poetry_section.get('version'),
            'description': poetry_section.get('description'),
            'authors': poetry_section.get('authors', []),
            'license': poetry_section.get('license'),
            'readme': poetry_section.get('readme'),
            'homepage': poetry_section.get('homepage'),
            'repository': poetry_section.get('repository'),
            'documentation': poetry_section.get('documentation'),
            'keywords': poetry_section.get('keywords', []),
            'classifiers': poetry_section.get('classifiers', []),
            'packages': poetry_section.get('packages', []),
            'include': poetry_section.get('include', []),
            'exclude': poetry_section.get('exclude', [])
        }
    
    def is_poetry_project(self) -> bool:
        """
        Check if the current directory is a Poetry project.
        
        Returns:
            True if pyproject.toml exists with Poetry configuration
        """
        if not os.path.exists(self.pyproject_path):
            return False
            
        try:
            with open(self.pyproject_path, 'r') as f:
                pyproject = toml.load(f)
            return 'poetry' in pyproject.get('tool', {})
        except:
            return False
    
    def get_virtualenv_path(self) -> Optional[str]:
        """
        Get the path to the current project's virtualenv.
        
        Returns:
            Path to virtualenv or None if not found
        """
        result = self._run_poetry_command(['env', 'info', '--path'])
        if result['success'] and result['stdout']:
            return result['stdout'].strip()
        return None
    
    def get_python_version(self) -> Optional[str]:
        """
        Get the Python version used by the project.
        
        Returns:
            Python version string or None
        """
        result = self.env_info()
        if result['success'] and result['stdout']:
            for line in result['stdout'].split('\n'):
                if 'Python' in line and 'version' in line.lower():
                    # Extract version from line like "Python:         3.9.7"
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[-1]
        return None
    
    def install_poetry(self) -> Dict[str, Any]:
        """
        Install Poetry if it's not already installed.
        
        Returns:
            Installation result
        """
        # Check if poetry is already installed
        check_result = subprocess.run(['which', 'poetry'], capture_output=True, text=True)
        if check_result.returncode == 0:
            return {
                'success': True,
                'message': 'Poetry is already installed',
                'path': check_result.stdout.strip()
            }
        
        # Install using the official installer
        install_cmd = 'curl -sSL https://install.python-poetry.org | python3 -'
        result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True)
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'message': 'Poetry installed successfully' if result.returncode == 0 else 'Failed to install Poetry'
        }
    
    def create_basic_project(self, name: str, 
                           description: str = "A Python project managed by Poetry",
                           author: Optional[str] = None,
                           python_version: str = "^3.8",
                           dependencies: Optional[Dict[str, str]] = None,
                           dev_dependencies: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Create a basic Poetry project with common structure.
        
        Args:
            name: Project name
            description: Project description
            author: Author name and email
            python_version: Python version requirement
            dependencies: Dictionary of dependencies
            dev_dependencies: Dictionary of dev dependencies
            
        Returns:
            Creation result
        """
        # Create project directory
        project_dir = os.path.join(self.project_path, name)
        os.makedirs(project_dir, exist_ok=True)
        
        # Create pyproject.toml
        pyproject_content = {
            'tool': {
                'poetry': {
                    'name': name,
                    'version': '0.1.0',
                    'description': description,
                    'authors': [author] if author else [],
                    'readme': 'README.md',
                    'dependencies': {
                        'python': python_version,
                        **(dependencies or {})
                    },
                    'dev-dependencies': dev_dependencies or {}
                }
            },
            'build-system': {
                'requires': ['poetry-core'],
                'build-backend': 'poetry.core.masonry.api'
            }
        }
        
        pyproject_path = os.path.join(project_dir, 'pyproject.toml')
        with open(pyproject_path, 'w') as f:
            toml.dump(pyproject_content, f)
        
        # Create README.md
        readme_path = os.path.join(project_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(f"# {name}\n\n{description}\n")
        
        # Create source directory
        src_dir = os.path.join(project_dir, name.replace('-', '_'))
        os.makedirs(src_dir, exist_ok=True)
        
        # Create __init__.py
        init_path = os.path.join(src_dir, '__init__.py')
        with open(init_path, 'w') as f:
            f.write(f'"""\n{description}\n"""\n\n__version__ = "0.1.0"\n')
        
        # Create tests directory
        tests_dir = os.path.join(project_dir, 'tests')
        os.makedirs(tests_dir, exist_ok=True)
        
        # Create test __init__.py
        test_init_path = os.path.join(tests_dir, '__init__.py')
        with open(test_init_path, 'w') as f:
            f.write('')
        
        return {
            'success': True,
            'project_path': project_dir,
            'message': f'Created project {name} at {project_dir}'
        }
    
    def forward(self, command: str, **kwargs) -> Dict[str, Any]:
        """
        Main entry point for executing Poetry commands.
        
        Args:
            command: Poetry command to execute
            **kwargs: Additional arguments for the command
            
        Returns:
            Command execution result
        """
        # Map command to appropriate method
        command_map = {
            'new': self.new,
            'init': self.init,
            'install': self.install,
            'add': self.add,
            'remove': self.remove,
            'update': self.update,
            'show': self.show,
            'build': self.build,
            'publish': self.publish,
            'config': self.config,
            'run': self.run,
            'shell': self.shell,
            'check': self.check,
            'search': self.search,
            'lock': self.lock,
            'export': self.export,
            'env': self.env_info,
            'version': self.version,
            'about': self.about,
            'source': self.source_show
        }
        
        if command in command_map:
            return command_map[command](**kwargs)
        else:
            # Fallback to direct command execution
            return self._run_poetry_command(command.split())
