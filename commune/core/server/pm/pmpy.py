#!/usr/bin/env python3
"""
PMPY - Python Native Package Manager
A simple Python-based package manager inspired by npm
"""

import os
import json
import subprocess
import sys
import shutil
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime
import commune as c


class PMPY:
    """
    Python Package Manager - A native Python implementation similar to npm
    """
    
    def __init__(self, project_dir: str = None):
        self.project_dir = project_dir or os.getcwd()
        self.package_file = os.path.join(self.project_dir, 'package.json')
        self.modules_dir = os.path.join(self.project_dir, 'python_modules')
        self.lock_file = os.path.join(self.project_dir, 'package-lock.json')
        
    def init(self, name: str = None, version: str = '1.0.0', description: str = '', 
             author: str = '', license: str = 'MIT') -> Dict[str, Any]:
        """
        Initialize a new Python package (similar to npm init)
        """
        if os.path.exists(self.package_file):
            return {'success': False, 'message': 'package.json already exists'}
            
        package_name = name or os.path.basename(self.project_dir)
        
        package_config = {
            'name': package_name,
            'version': version,
            'description': description,
            'author': author,
            'license': license,
            'dependencies': {},
            'devDependencies': {},
            'scripts': {
                'test': 'python -m pytest',
                'start': 'python main.py'
            },
            'python': {
                'version': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
            }
        }
        
        with open(self.package_file, 'w') as f:
            json.dump(package_config, f, indent=2)
            
        c.print(f'Initialized {package_name} package', color='green')
        return {'success': True, 'package': package_config}
        
    def install(self, package: str = None, save: bool = True, dev: bool = False) -> Dict[str, Any]:
        """
        Install a Python package (similar to npm install)
        """
        if not os.path.exists(self.package_file) and package is None:
            return {'success': False, 'message': 'No package.json found. Run pmpy init first.'}
            
        # Create modules directory if it doesn't exist
        os.makedirs(self.modules_dir, exist_ok=True)
        
        if package is None:
            # Install all dependencies from package.json
            return self._install_all_dependencies()
        else:
            # Install specific package
            return self._install_package(package, save, dev)
            
    def _install_package(self, package: str, save: bool, dev: bool) -> Dict[str, Any]:
        """
        Install a specific package
        """
        try:
            # Parse package name and version
            if '==' in package:
                pkg_name, version = package.split('==')
            else:
                pkg_name = package
                version = None
                
            # Install using pip to local modules directory
            cmd = [sys.executable, '-m', 'pip', 'install']
            if version:
                cmd.append(f'{pkg_name}=={version}')
            else:
                cmd.append(pkg_name)
            cmd.extend(['--target', self.modules_dir])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {'success': False, 'message': result.stderr}
                
            # Update package.json if save is True
            if save and os.path.exists(self.package_file):
                with open(self.package_file, 'r') as f:
                    package_data = json.load(f)
                    
                # Get installed version
                installed_version = self._get_installed_version(pkg_name)
                
                if dev:
                    package_data['devDependencies'][pkg_name] = installed_version or 'latest'
                else:
                    package_data['dependencies'][pkg_name] = installed_version or 'latest'
                    
                with open(self.package_file, 'w') as f:
                    json.dump(package_data, f, indent=2)
                    
            c.print(f'Successfully installed {package}', color='green')
            return {'success': True, 'package': pkg_name, 'version': installed_version}
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
            
    def _install_all_dependencies(self) -> Dict[str, Any]:
        """
        Install all dependencies from package.json
        """
        with open(self.package_file, 'r') as f:
            package_data = json.load(f)
            
        all_deps = {}
        all_deps.update(package_data.get('dependencies', {}))
        all_deps.update(package_data.get('devDependencies', {}))
        
        installed = []
        failed = []
        
        for pkg_name, version in all_deps.items():
            if version == 'latest':
                package_spec = pkg_name
            else:
                package_spec = f'{pkg_name}=={version}'
                
            result = self._install_package(package_spec, save=False, dev=False)
            
            if result['success']:
                installed.append(pkg_name)
            else:
                failed.append(pkg_name)
                
        return {
            'success': len(failed) == 0,
            'installed': installed,
            'failed': failed
        }
        
    def uninstall(self, package: str, save: bool = True) -> Dict[str, Any]:
        """
        Uninstall a package (similar to npm uninstall)
        """
        try:
            # Remove from modules directory
            package_path = os.path.join(self.modules_dir, package)
            if os.path.exists(package_path):
                shutil.rmtree(package_path)
                
            # Update package.json if save is True
            if save and os.path.exists(self.package_file):
                with open(self.package_file, 'r') as f:
                    package_data = json.load(f)
                    
                package_data['dependencies'].pop(package, None)
                package_data['devDependencies'].pop(package, None)
                
                with open(self.package_file, 'w') as f:
                    json.dump(package_data, f, indent=2)
                    
            c.print(f'Successfully uninstalled {package}', color='green')
            return {'success': True, 'package': package}
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
            
    def list(self, depth: int = 0) -> Dict[str, Any]:
        """
        List installed packages (similar to npm list)
        """
        if not os.path.exists(self.package_file):
            return {'success': False, 'message': 'No package.json found'}
            
        with open(self.package_file, 'r') as f:
            package_data = json.load(f)
            
        installed_packages = self._get_installed_packages()
        
        return {
            'success': True,
            'name': package_data.get('name', 'unknown'),
            'version': package_data.get('version', '0.0.0'),
            'dependencies': package_data.get('dependencies', {}),
            'devDependencies': package_data.get('devDependencies', {}),
            'installed': installed_packages
        }
        
    def run(self, script: str) -> Dict[str, Any]:
        """
        Run a script defined in package.json (similar to npm run)
        """
        if not os.path.exists(self.package_file):
            return {'success': False, 'message': 'No package.json found'}
            
        with open(self.package_file, 'r') as f:
            package_data = json.load(f)
            
        scripts = package_data.get('scripts', {})
        
        if script not in scripts:
            return {'success': False, 'message': f'Script "{script}" not found'}
            
        cmd = scripts[script]
        
        # Add python_modules to PYTHONPATH
        env = os.environ.copy()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{self.modules_dir}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = self.modules_dir
            
        try:
            c.print(f'Running script: {script}', color='cyan')
            result = subprocess.run(cmd, shell=True, env=env)
            return {'success': result.returncode == 0, 'script': script, 'command': cmd}
        except Exception as e:
            return {'success': False, 'message': str(e)}
            
    def _get_installed_version(self, package: str) -> Optional[str]:
        """
        Get the installed version of a package
        """
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', package],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        return line.split(':')[1].strip()
                        
        except Exception:
            pass
            
        return None
        
    def _get_installed_packages(self) -> List[str]:
        """
        Get list of installed packages in python_modules
        """
        if not os.path.exists(self.modules_dir):
            return []
            
        packages = []
        for item in os.listdir(self.modules_dir):
            if os.path.isdir(os.path.join(self.modules_dir, item)) and not item.startswith('.'):
                packages.append(item)
                
        return packages
        
    def forward(self, action: str = 'help', **kwargs) -> Any:
        """
        Main interface for PMPY commands
        """
        actions = {
            'init': self.init,
            'install': self.install,
            'uninstall': self.uninstall,
            'list': self.list,
            'run': self.run,
            'help': self.help
        }
        
        if action in actions:
            return actions[action](**kwargs)
        else:
            return {'success': False, 'message': f'Unknown action: {action}'}
            
    def help(self) -> Dict[str, Any]:
        """
        Show help information
        """
        help_text = """
PMPY - Python Package Manager

Commands:
  init              Initialize a new package.json file
  install [pkg]     Install a package or all dependencies
  uninstall [pkg]   Remove a package
  list              List installed packages
  run [script]      Run a script from package.json
  help              Show this help message

Examples:
  pmpy init
  pmpy install requests
  pmpy install requests --save-dev
  pmpy run test
        """
        c.print(help_text, color='cyan')
        return {'success': True, 'message': 'Help displayed'}


def main():
    """
    CLI entry point for PMPY
    """
    parser = argparse.ArgumentParser(description='PMPY - Python Package Manager')
    parser.add_argument('command', help='Command to run', nargs='?', default='help')
    parser.add_argument('package', help='Package name', nargs='?')
    parser.add_argument('--save', action='store_true', help='Save to dependencies')
    parser.add_argument('--save-dev', action='store_true', help='Save to devDependencies')
    parser.add_argument('--name', help='Package name for init')
    parser.add_argument('--version', help='Package version for init', default='1.0.0')
    parser.add_argument('--description', help='Package description for init', default='')
    parser.add_argument('--author', help='Package author for init', default='')
    parser.add_argument('--license', help='Package license for init', default='MIT')
    
    args = parser.parse_args()
    
    pmpy = PMPY()
    
    if args.command == 'init':
        result = pmpy.init(
            name=args.name,
            version=args.version,
            description=args.description,
            author=args.author,
            license=args.license
        )
    elif args.command == 'install':
        result = pmpy.install(
            package=args.package,
            save=args.save or not args.save_dev,
            dev=args.save_dev
        )
    elif args.command == 'uninstall':
        result = pmpy.uninstall(package=args.package, save=args.save)
    elif args.command == 'list':
        result = pmpy.list()
        if result['success']:
            c.print(f"\n{result['name']}@{result['version']}", color='green')
            if result['dependencies']:
                c.print("\nDependencies:", color='cyan')
                for pkg, ver in result['dependencies'].items():
                    c.print(f"  {pkg}: {ver}")
            if result['devDependencies']:
                c.print("\nDev Dependencies:", color='cyan')
                for pkg, ver in result['devDependencies'].items():
                    c.print(f"  {pkg}: {ver}")
    elif args.command == 'run':
        if args.package:
            result = pmpy.run(script=args.package)
        else:
            result = {'success': False, 'message': 'Please specify a script to run'}
    else:
        result = pmpy.help()
    
    if not result['success'] and 'message' in result:
        c.print(f"Error: {result['message']}", color='red')
        sys.exit(1)


if __name__ == '__main__':
    main()
