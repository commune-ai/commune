#!/usr/bin/env python3
"""
Test suite for the Poetry module.

This test suite verifies the functionality of the Poetry module.
"""

import os
import tempfile
import shutil
import commune as c


class TestPoetry:
    """
    Test cases for the Poetry module.
    """
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.poetry = c.module('poetry')(project_path=self.temp_dir)
        
    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_poetry_installation_check(self):
        """Test checking if Poetry is installed."""
        result = self.poetry._run_poetry_command(['--version'])
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'command' in result
    
    def test_is_poetry_project(self):
        """Test checking if a directory is a Poetry project."""
        # Should be False for empty directory
        assert not self.poetry.is_poetry_project()
        
        # Create a pyproject.toml with Poetry configuration
        pyproject_content = '''
[tool.poetry]
name = "test-project"
version = "0.1.0"
description = "Test project"
authors = ["Test Author <test@example.com>"]

[tool.poetry.dependencies]
python = "^3.8"
        '''
        
        with open(os.path.join(self.temp_dir, 'pyproject.toml'), 'w') as f:
            f.write(pyproject_content)
        
        # Should now be True
        assert self.poetry.is_poetry_project()
    
    def test_create_basic_project(self):
        """Test creating a basic Poetry project."""
        result = self.poetry.create_basic_project(
            name='test-project',
            description='A test project',
            author='Test Author <test@example.com>',
            dependencies={'requests': '^2.31.0'},
            dev_dependencies={'pytest': '^7.4.0'}
        )
        
        assert result['success']
        assert 'project_path' in result
        
        # Check that files were created
        project_path = result['project_path']
        assert os.path.exists(os.path.join(project_path, 'pyproject.toml'))
        assert os.path.exists(os.path.join(project_path, 'README.md'))
        assert os.path.exists(os.path.join(project_path, 'test_project', '__init__.py'))
        assert os.path.exists(os.path.join(project_path, 'tests', '__init__.py'))
    
    def test_get_project_info(self):
        """Test getting project information."""
        # Create a project first
        self.poetry.create_basic_project(
            name='info-test',
            description='Testing project info',
            author='Info Tester <info@test.com>'
        )
        
        # Update poetry instance to point to the new project
        project_poetry = c.module('poetry')(
            project_path=os.path.join(self.temp_dir, 'info-test')
        )
        
        info = project_poetry.get_project_info()
        assert info['name'] == 'info-test'
        assert info['version'] == '0.1.0'
        assert info['description'] == 'Testing project info'
        assert 'Info Tester <info@test.com>' in info['authors']
    
    def test_get_dependencies(self):
        """Test getting dependencies from pyproject.toml."""
        # Create a project with dependencies
        self.poetry.create_basic_project(
            name='deps-test',
            dependencies={'requests': '^2.31.0', 'click': '^8.0.0'},
            dev_dependencies={'pytest': '^7.4.0'}
        )
        
        # Update poetry instance
        project_poetry = c.module('poetry')(
            project_path=os.path.join(self.temp_dir, 'deps-test')
        )
        
        # Get regular dependencies
        deps = project_poetry.get_dependencies()
        assert 'requests' in deps
        assert deps['requests'] == '^2.31.0'
        assert 'click' in deps
        
        # Get dev dependencies
        dev_deps = project_poetry.get_dependencies(dev=True)
        assert 'pytest' in dev_deps
        assert dev_deps['pytest'] == '^7.4.0'
    
    def test_command_execution(self):
        """Test executing Poetry commands."""
        # Test a simple command
        result = self.poetry._run_poetry_command(['--version'])
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'stdout' in result or 'error' in result
        
        # Test with invalid command
        result = self.poetry._run_poetry_command(['invalid-command'])
        assert not result['success']
    
    def test_forward_method(self):
        """Test the forward method for command routing."""
        # Create a test project
        self.poetry.create_basic_project(name='forward-test')
        project_poetry = c.module('poetry')(
            project_path=os.path.join(self.temp_dir, 'forward-test')
        )
        
        # Test various commands through forward
        result = project_poetry.forward('check')
        assert isinstance(result, dict)
        assert 'success' in result
        
        # Test with unknown command (should fall back to direct execution)
        result = project_poetry.forward('--version')
        assert isinstance(result, dict)


def run_tests():
    """Run all tests and report results."""
    print("Running Poetry Module Tests")
    print("=" * 50)
    
    test_instance = TestPoetry()
    test_methods = [
        method for method in dir(test_instance) 
        if method.startswith('test_')
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        print(f"\nRunning {test_method}...")
        try:
            test_instance.setup_method()
            getattr(test_instance, test_method)()
            test_instance.teardown_method()
            print(f"✓ {test_method} passed")
            passed += 1
        except Exception as e:
            print(f"✗ {test_method} failed: {str(e)}")
            failed += 1
            try:
                test_instance.teardown_method()
            except:
                pass
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
