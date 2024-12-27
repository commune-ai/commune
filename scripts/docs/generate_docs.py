#!/usr/bin/env python3
"""
Documentation Generation Script

This script automates the generation of documentation for the Commune library:
1. Extracts docstrings and type hints from code
2. Generates API reference documentation
3. Updates markdown files with latest examples
4. Validates documentation links and references
"""

import os
import sys
import glob
import inspect
import logging
from typing import List, Dict, Any
import commune as c

class DocGenerator:
    def __init__(self):
        self.root_dir = c.repo_path
        self.wiki_dir = os.path.join(self.root_dir, 'wiki')
        self.docs_dir = os.path.join(self.root_dir, 'docs')
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for documentation generation."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('doc_generation.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def generate_api_docs(self):
        """Generate API documentation from code."""
        self.logger.info("Generating API documentation...")
        
        modules = self.discover_modules()
        for module_path in modules:
            try:
                module_docs = self.extract_module_docs(module_path)
                self.write_module_docs(module_path, module_docs)
            except Exception as e:
                self.logger.error(f"Error processing {module_path}: {str(e)}")

    def discover_modules(self) -> List[str]:
        """Find all Python modules in the codebase."""
        module_paths = []
        for root, _, files in os.walk(os.path.join(self.root_dir, 'commune')):
            for file in files:
                if file.endswith('.py') and not file.startswith('_'):
                    module_paths.append(os.path.join(root, file))
        return module_paths

    def extract_module_docs(self, module_path: str) -> Dict[str, Any]:
        """Extract documentation from a module."""
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        module = c.import_module(module_path)
        
        docs = {
            'name': module_name,
            'doc': inspect.getdoc(module) or '',
            'classes': {},
            'functions': {}
        }

        for name, obj in inspect.getmembers(module):
            if name.startswith('_'):
                continue

            if inspect.isclass(obj):
                docs['classes'][name] = self.extract_class_docs(obj)
            elif inspect.isfunction(obj):
                docs['functions'][name] = self.extract_function_docs(obj)

        return docs

    def extract_class_docs(self, cls) -> Dict[str, Any]:
        """Extract documentation from a class."""
        return {
            'doc': inspect.getdoc(cls) or '',
            'methods': {
                name: self.extract_function_docs(method)
                for name, method in inspect.getmembers(cls)
                if inspect.isfunction(method) and not name.startswith('_')
            }
        }

    def extract_function_docs(self, func) -> Dict[str, Any]:
        """Extract documentation from a function."""
        return {
            'doc': inspect.getdoc(func) or '',
            'signature': str(inspect.signature(func)),
            'annotations': {
                k: str(v) for k, v in func.__annotations__.items()
            }
        }

    def write_module_docs(self, module_path: str, docs: Dict[str, Any]):
        """Write module documentation to markdown file."""
        relative_path = os.path.relpath(module_path, self.root_dir)
        doc_path = os.path.join(
            self.docs_dir,
            'api',
            relative_path.replace('.py', '.md')
        )

        os.makedirs(os.path.dirname(doc_path), exist_ok=True)

        with open(doc_path, 'w') as f:
            f.write(f"# {docs['name']}\n\n")
            f.write(f"{docs['doc']}\n\n")

            if docs['classes']:
                f.write("## Classes\n\n")
                for name, class_docs in docs['classes'].items():
                    f.write(f"### {name}\n\n")
                    f.write(f"{class_docs['doc']}\n\n")
                    
                    if class_docs['methods']:
                        f.write("#### Methods\n\n")
                        for method_name, method_docs in class_docs['methods'].items():
                            f.write(f"##### `{method_name}{method_docs['signature']}`\n\n")
                            f.write(f"{method_docs['doc']}\n\n")

            if docs['functions']:
                f.write("## Functions\n\n")
                for name, func_docs in docs['functions'].items():
                    f.write(f"### `{name}{func_docs['signature']}`\n\n")
                    f.write(f"{func_docs['doc']}\n\n")

    def update_examples(self):
        """Update code examples in documentation."""
        self.logger.info("Updating code examples...")
        
        example_files = glob.glob(
            os.path.join(self.wiki_dir, '*.md')
        )
        
        for file_path in example_files:
            try:
                self.update_file_examples(file_path)
            except Exception as e:
                self.logger.error(f"Error updating {file_path}: {str(e)}")

    def update_file_examples(self, file_path: str):
        """Update code examples in a single file."""
        with open(file_path, 'r') as f:
            content = f.read()

        # Update example outputs
        # This is a placeholder - implement based on your needs
        updated_content = content

        with open(file_path, 'w') as f:
            f.write(updated_content)

    def validate_docs(self):
        """Validate documentation files."""
        self.logger.info("Validating documentation...")
        
        issues = []
        
        # Check for broken links
        issues.extend(self.check_links())
        
        # Check for outdated content
        issues.extend(self.check_content_freshness())
        
        # Report issues
        if issues:
            self.logger.warning("Documentation issues found:")
            for issue in issues:
                self.logger.warning(f"- {issue}")
        else:
            self.logger.info("No documentation issues found")

    def check_links(self) -> List[str]:
        """Check for broken links in documentation."""
        issues = []
        markdown_files = glob.glob(
            os.path.join(self.wiki_dir, '**/*.md'),
            recursive=True
        )
        
        for file_path in markdown_files:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check internal links
            for link in self.extract_links(content):
                if not self.validate_link(link):
                    issues.append(f"Broken link in {file_path}: {link}")
        
        return issues

    def extract_links(self, content: str) -> List[str]:
        """Extract markdown links from content."""
        # This is a simple implementation - enhance based on your needs
        import re
        return re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', content)

    def validate_link(self, link: str) -> bool:
        """Validate if a link is valid."""
        if link.startswith(('http://', 'https://')):
            # Skip external links for now
            return True
        
        # Check if internal link target exists
        target_path = os.path.join(self.root_dir, link)
        return os.path.exists(target_path)

    def check_content_freshness(self) -> List[str]:
        """Check for outdated content."""
        issues = []
        threshold = 90 * 24 * 60 * 60  # 90 days in seconds
        
        for root, _, files in os.walk(self.wiki_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    age = c.time() - os.path.getmtime(file_path)
                    
                    if age > threshold:
                        issues.append(
                            f"{file_path} is {age//(24*60*60)} days old"
                        )
        
        return issues

def main():
    """Main entry point for documentation generation."""
    generator = DocGenerator()
    
    try:
        # Generate API documentation
        generator.generate_api_docs()
        
        # Update examples
        generator.update_examples()
        
        # Validate documentation
        generator.validate_docs()
        
        generator.logger.info("Documentation generation complete")
        return 0
        
    except Exception as e:
        generator.logger.error(f"Documentation generation failed: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
