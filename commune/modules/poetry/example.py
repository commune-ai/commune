#!/usr/bin/env python3
"""
Example usage of the Poetry module.

This script demonstrates various features of the Poetry module.
"""

import os
import commune as c


def main():
    # Initialize the Poetry module
    poetry = c.module('poetry')()
    
    print("Poetry Module Example")
    print("=" * 50)
    
    # Check if Poetry is installed
    print("\n1. Checking Poetry installation...")
    check_result = poetry._run_poetry_command(['--version'])
    if check_result['success']:
        print(f"✓ Poetry is installed: {check_result['stdout'].strip()}")
    else:
        print("✗ Poetry is not installed. Installing...")
        install_result = poetry.install_poetry()
        print(install_result['message'])
    
    # Check if current directory is a Poetry project
    print("\n2. Checking current directory...")
    if poetry.is_poetry_project():
        print("✓ Current directory is a Poetry project")
        
        # Get project info
        project_info = poetry.get_project_info()
        print(f"\nProject: {project_info['name']} v{project_info['version']}")
        print(f"Description: {project_info['description']}")
        
        # Show dependencies
        print("\n3. Current dependencies:")
        deps = poetry.get_dependencies()
        for pkg, version in deps.items():
            print(f"  - {pkg}: {version}")
        
        # Show dev dependencies
        dev_deps = poetry.get_dependencies(dev=True)
        if dev_deps:
            print("\nDev dependencies:")
            for pkg, version in dev_deps.items():
                print(f"  - {pkg}: {version}")
    else:
        print("✗ Current directory is not a Poetry project")
        
        # Offer to create a new project
        print("\nWould you like to create a sample project? (y/n)")
        response = input("> ").lower().strip()
        
        if response == 'y':
            print("\n4. Creating a sample project...")
            result = poetry.create_basic_project(
                name='poetry-example',
                description='Example project demonstrating Poetry module',
                author='Example Author <author@example.com>',
                dependencies={
                    'requests': '^2.31.0',
                    'click': '^8.1.0'
                },
                dev_dependencies={
                    'pytest': '^7.4.0',
                    'black': '^23.0.0'
                }
            )
            print(f"✓ {result['message']}")
            
            # Change to the new project directory
            os.chdir(result['project_path'])
            poetry = c.module('poetry')(project_path=result['project_path'])
            
            print("\n5. Installing dependencies...")
            install_result = poetry.install()
            if install_result['success']:
                print("✓ Dependencies installed successfully")
            else:
                print(f"✗ Failed to install dependencies: {install_result.get('error', 'Unknown error')}")
    
    # Demonstrate some common operations
    print("\n" + "=" * 50)
    print("Common Poetry Operations:")
    print("=" * 50)
    
    operations = [
        ("Check project validity", lambda: poetry.check()),
        ("Show environment info", lambda: poetry.env_info()),
        ("List installed packages", lambda: poetry.show(top_level=True)),
        ("Show outdated packages", lambda: poetry.show(outdated=True))
    ]
    
    for op_name, op_func in operations:
        print(f"\n{op_name}:")
        result = op_func()
        if result['success']:
            output = result['stdout'].strip()
            if output:
                # Limit output to first 5 lines for brevity
                lines = output.split('\n')[:5]
                for line in lines:
                    print(f"  {line}")
                if len(output.split('\n')) > 5:
                    print("  ...")
        else:
            print(f"  ✗ Operation failed: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nYou can now use the Poetry module for:")
    print("  - Managing dependencies (add, remove, update)")
    print("  - Building and publishing packages")
    print("  - Managing virtual environments")
    print("  - Running scripts and commands")
    print("  - And much more!")
    print("\nSee the README.md for full documentation.")


if __name__ == "__main__":
    main()
