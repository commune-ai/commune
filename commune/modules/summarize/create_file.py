
import commune as c
import os
from typing import Dict, Any, Optional, Union
from ..utils import abspath, put_text, ensure_directory_exists

class CreateFile:
    """
    A utility tool for creating new files at specified paths.
    
    This class provides functionality to:
    - Create new files with specified content
    - Ensure parent directories exist
    - Handle different file types appropriately
    - Provide feedback on the operation
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the CreateFile tool.
        
        Args:
            **kwargs: Additional configuration parameters
        """
        pass
    
    def forward(self, 
                file_path: str, 
                content: str = "",
                create_parent_dirs: bool = True,
                overwrite: bool = False,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Create a new file at the specified path with the given content.
        
        Args:
            file_path: Path where the file should be created
            content: Content to write to the file
            create_parent_dirs: Whether to create parent directories if they don't exist
            overwrite: Whether to overwrite the file if it already exists
            verbose: Print detailed information about the operation
            
        Returns:
            Dictionary with operation results including:
            - success: Whether the operation was successful
            - file_path: Path to the created file
            - message: Description of the operation result
        """
        file_path = abspath(file_path)
        # Check if file already exists
        if os.path.exists(file_path) and not overwrite:
            return {
                "success": False,
                "file_path": file_path,
                "message": f"File already exists and overwrite is False"
            }
        
        # Create parent directories if needed
        parent_dir = os.path.dirname(file_path)
        if create_parent_dirs and parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            if verbose:
                c.print(f"Created parent directory: {parent_dir}", color="green")
            
        put_text(file_path, content)
        return {
            "success": True,
            "file_path": file_path,
            "message": f"File created successfully at {file_path}"
        }
    
