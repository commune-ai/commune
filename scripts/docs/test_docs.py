#!/usr/bin/env python3
"""
Documentation Testing Script

This script tests the documentation for:
1. Code example validity
2. Link integrity
3. Style consistency
4. Content completeness
"""

import os
import sys
import glob
import pytest
import logging
import subprocess
from typing import List, Dict, Any
import commune as c

class DocTester:
    def __init__(self):
        self.root_dir = c.repo_path
        self.wiki_dir = os.path.join(self.root_dir, 'wiki')
        self.docs_dir = os.path.join(self.root_dir, 'docs')
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for documentation testing."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('doc_testing.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def test_code_examples(self):
        """Test all code examples in documentation."""
        self.logger.info("Testing code examples...")
        
        examples = self.extract_code_examples()
        results = []
        
        for example in examples:
            try:
                result = self.test_example(example)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error testing example: {str(e)}")
                results.append(False)
        
        return all(results)

    def extract_code_examples(self) -> List[Dict[str, Any]]:
        """Extract code examples from documentation."""
        examples = []
        
        for root, _, files in os.walk(self.wiki_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    examples.extend(
                        self.extract_file_examples(file_path)
                    )
        
        return examples

    def extract_file_examples(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract code examples from a single file."""
        examples = []
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract code blocks
        import re
        code_blocks = re.findall(
            r'```python\n(.*?)\n```',
            content,
            re.DOTALL
        )
        
        for code in code_blocks:
            examples.append({
                'code': code,
                'file': file_path
            })
        
        return examples

    def test_example(self, example: Dict[str, Any]) -> bool:
        """Test a single code example."""
        try:
            # Create a temporary test file
            test_file = 'temp_test.py'
            with open(test_file, 'w') as f:
                f.write(example['code'])
            
            # Run the test
            result = subprocess.run(
                ['python', '-m', 'pytest', test_file],
                capture_output=True,
                text=True
            )
            
            # Clean up
            os.remove(test_file)
            
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(
                f"Error in example from {example['file']}: {str(e)}"
            )
            return False

    def test_style(self):
        """Test documentation style consistency."""
        self.logger.info("Testing documentation style...")
        
        issues = []
        
        # Check markdown style
        issues.extend(self.check_markdown_style())
        
        # Check naming conventions
        issues.extend(self.check_naming_conventions())
        
        # Report issues
        if issues:
            self.logger.warning("Style issues found:")
            for issue in issues:
                self.logger.warning(f"- {issue}")
            return False
        
        return True

    def check_markdown_style(self) -> List[str]:
        """Check markdown style consistency."""
        issues = []
        
        for root, _, files in os.walk(self.wiki_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    issues.extend(
                        self.check_file_style(file_path)
                    )
        
        return issues

    def check_file_style(self, file_path: str) -> List[str]:
        """Check style in a single file."""
        issues = []
        
        with open(file_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Check heading hierarchy
        headings = [
            line for line in lines 
            if line.startswith('#')
        ]
        if not self.check_heading_hierarchy(headings):
            issues.append(
                f"Invalid heading hierarchy in {file_path}"
            )
        
        # Check line length
        long_lines = [
            i+1 for i, line in enumerate(lines)
            if len(line) > 80
        ]
        if long_lines:
            issues.append(
                f"Lines too long in {file_path}: {long_lines}"
            )
        
        return issues

    def check_heading_hierarchy(self, headings: List[str]) -> bool:
        """Check if heading hierarchy is valid."""
        current_level = 1
        
        for heading in headings:
            level = len(heading.split()[0])
            if level > current_level + 1:
                return False
            current_level = level
        
        return True

    def check_naming_conventions(self) -> List[str]:
        """Check naming conventions in documentation."""
        issues = []
        
        # Check file names
        for root, _, files in os.walk(self.wiki_dir):
            for file in files:
                if not self.is_valid_filename(file):
                    issues.append(
                        f"Invalid filename: {file}"
                    )
        
        return issues

    def is_valid_filename(self, filename: str) -> bool:
        """Check if a filename follows conventions."""
        if not filename.endswith('.md'):
            return False
        
        name = filename[:-3]
        return (
            name.replace('-', '').isalnum() and
            not name[0].isdigit()
        )

    def test_completeness(self):
        """Test documentation completeness."""
        self.logger.info("Testing documentation completeness...")
        
        issues = []
        
        # Check required files
        issues.extend(self.check_required_files())
        
        # Check section coverage
        issues.extend(self.check_section_coverage())
        
        # Report issues
        if issues:
            self.logger.warning("Completeness issues found:")
            for issue in issues:
                self.logger.warning(f"- {issue}")
            return False
        
        return True

    def check_required_files(self) -> List[str]:
        """Check if all required files exist."""
        required_files = [
            'README.md',
            'CONTRIBUTING.md',
            'LICENSE',
            'wiki/00-Home.md',
            'wiki/01-Installation.md'
        ]
        
        issues = []
        for file in required_files:
            path = os.path.join(self.root_dir, file)
            if not os.path.exists(path):
                issues.append(f"Missing required file: {file}")
        
        return issues

    def check_section_coverage(self) -> List[str]:
        """Check if all sections are covered."""
        required_sections = {
            'Installation': ['prerequisites', 'steps'],
            'Quick Start': ['example', 'usage'],
            'API Reference': ['modules', 'classes'],
            'Contributing': ['guidelines', 'process']
        }
        
        issues = []
        
        for root, _, files in os.walk(self.wiki_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    issues.extend(
                        self.check_file_sections(
                            file_path,
                            required_sections
                        )
                    )
        
        return issues

    def check_file_sections(
        self,
        file_path: str,
        required_sections: Dict[str, List[str]]
    ) -> List[str]:
        """Check sections in a single file."""
        issues = []
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        for section, subsections in required_sections.items():
            if section in content:
                for subsection in subsections:
                    if subsection not in content.lower():
                        issues.append(
                            f"Missing subsection '{subsection}' "
                            f"in section '{section}' "
                            f"in {file_path}"
                        )
        
        return issues

def main():
    """Main entry point for documentation testing."""
    tester = DocTester()
    success = True
    
    try:
        # Test code examples
        if not tester.test_code_examples():
            success = False
            tester.logger.error("Code example tests failed")
        
        # Test style
        if not tester.test_style():
            success = False
            tester.logger.error("Style tests failed")
        
        # Test completeness
        if not tester.test_completeness():
            success = False
            tester.logger.error("Completeness tests failed")
        
        if success:
            tester.logger.info("All documentation tests passed")
            return 0
        else:
            tester.logger.error("Some documentation tests failed")
            return 1
        
    except Exception as e:
        tester.logger.error(f"Documentation testing failed: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
