import commune as c


class Torus(c.Module):
    """
    A class for commune that provides Torus functionality.
    Includes methods for context extraction and README processing.
    """
    
    def __init__(self, core_path=None):
        super().__init__()
        self.core_path = core_path or self.resolve_path('core')
    
    def context(self, path=None):
        """
        Extract and return README content from the specified path.
        
        Args:
            path: Path to search for README files. Defaults to core_path.
            
        Returns:
            str: The extracted README content as text.
        """
        path = path or self.core_path
        readme2text = self.readme2text(path)
        print('ctx size', len(str(readme2text)))
        return readme2text
    
    def readme2text(self, path):
        """
        Convert README files to text format.
        
        Args:
            path: Path to the README file or directory containing README files.
            
        Returns:
            str: The README content as plain text.
        """
        # Implementation to extract README content
        import os
        
        readme_files = ['README.md', 'README.txt', 'README.rst', 'README']
        readme_content = []
        
        if os.path.isfile(path):
            # If path is a file, read it directly
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif os.path.isdir(path):
            # If path is a directory, look for README files
            for readme_name in readme_files:
                readme_path = os.path.join(path, readme_name)
                if os.path.exists(readme_path):
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        readme_content.append(f.read())
            
            # Also search subdirectories for README files
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file in readme_files:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            readme_content.append(f"\n\n--- {file_path} ---\n\n{f.read()}")
        
        return '\n'.join(readme_content) if readme_content else 'No README files found.'
    
    def resolve_path(self, path):
        """
        Resolve a path relative to the module's location.
        
        Args:
            path: Relative path to resolve.
            
        Returns:
            str: Absolute path.
        """
        import os
        module_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(module_dir, path)
