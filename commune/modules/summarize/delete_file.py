
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
        path = abspath(path)

        assert isinstance(path, str), f"Path should be a string, got {type(path)}"
        
        assert os.path.exists(path), f"Path does not exist: {path}. Set force=True to ignore this error."
        assert os.path.isfile(path), f"Path is neither a file nor a directory: {path}"

        # Check if path exists
        if not os.path.exists(path):
            if force:
                if verbose:
                    c.print(f"Path does not exist, but force=True: {path}", color="yellow")
                return {
                    "success": True,
                    "path": path,
                    "message": "Path does not exist, but operation considered successful due to force=True"
                }
            else:
                if verbose:
                    c.print(f"Path does not exist: {path}", color="red")
                return {
                    "success": False,
                    "path": path,
                    "message": "Path does not exist"
                }
        
        # Handle directory deletion
        if os.path.isdir(path):
            if not allow_dir:
                if verbose:
                    c.print(f"Path is a directory but allow_dir=False: {path}", color="red")
                return {
                    "success": False,
                    "path": path,
                    "message": "Path is a directory but allow_dir=False"
                }
            
            try:
                if recursive:
                    shutil.rmtree(path)
                    if verbose:
                        c.print(f"Successfully deleted directory recursively: {path}", color="green")
                    return {
                        "success": True,
                        "path": path,
                        "message": "Directory deleted recursively"
                    }
                else:
                    os.rmdir(path)
                    if verbose:
                        c.print(f"Successfully deleted empty directory: {path}", color="green")
                    return {
                        "success": True,
                        "path": path,
                        "message": "Empty directory deleted"
                    }
            except OSError as e:
                if not recursive and len(os.listdir(path)) > 0:
                    if verbose:
                        c.print(f"Cannot delete non-empty directory without recursive=True: {path}", color="red")
                    return {
                        "success": False,
                        "path": path,
                        "message": "Cannot delete non-empty directory without recursive=True"
                    }
                if verbose:
                    c.print(f"Failed to delete directory: {str(e)}", color="red")
                return {
                    "success": False,
                    "path": path,
                    "message": f"Failed to delete directory: {str(e)}"
                }
        
        # Handle file deletion
        try:
            os.remove(path)
            if verbose:
                c.print(f"Successfully deleted file: {path}", color="green")
            return {
                "success": True,
                "path": path,
                "message": "File deleted successfully"
            }
        except Exception as e:
            if verbose:
                c.print(f"Failed to delete file: {str(e)}", color="red")
            return {
                "success": False,
                "path": path,
                "message": f"Failed to delete file: {str(e)}"
            }
