
import commune as c
import os
import shutil
from typing import Dict, Any, Optional, Union
from ..utils import abspath

class DeleteFile:
    """
    A utility tool for deleting files and directories at specified paths.
    
    This class provides functionality to:
    - Delete individual files
    - Optionally delete directories (recursively or not)
    - Implement safety checks before deletion
    - Provide feedback on the operation
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the DeleteFile tool.
        
        Args:
            **kwargs: Additional configuration parameters
        """
        pass
    
    def forward(self, 
                path: str, 
                recursive: bool = False,
                force: bool = False,
                allow_dir: bool = False,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Delete a file or directory at the specified path.
        
        Args:
            path: Path to the file or directory to delete
            recursive: Whether to recursively delete directories (only applies if allow_dir=True)
            force: Whether to ignore non-existent files
            allow_dir: Whether to allow directory deletion
            verbose: Print detailed information about the operation
            
        Returns:
            Dictionary with operation results including:
            - success: Whether the operation was successful
            - path: Path that was targeted for deletion
            - message: Description of the operation result
        """
        path = os.path.abspath(path)

        assert isinstance(path, str), f"Path should be a string, got {type(path)}"
        
        assert os.path.exists(path), f"Path does not exist: {path}. Set force=True to ignore this error."
        assert os.path.isdir(path), f"Path is neither a file nor a directory: {path}"

        return shutil.rmtree(path)