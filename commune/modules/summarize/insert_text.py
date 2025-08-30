
import commune as c
import os
import re
from typing import Dict, List, Union, Optional, Any
from ..utils import get_text, put_text, abspath
print = c.print
class Insert:
    """
    A utility tool for inserting text between specified anchor points in files.
    
    This class provides functionality to:
    - Find anchor points in files
    - Insert new content between those anchor points
    - Replace existing content between anchor points
    - Preserve the original file structure
    - Handle multiple insertions in a single operation
    """
    
    def __init__(self, cache_dir: str = '~/.commune/insert_cache', **kwargs):
        """
        Initialize the Insert tool.
        
        Args:
            cache_dir: Directory to store cache files if needed
            **kwargs: Additional configuration parameters
        """
        self.cache_dir = abspath(cache_dir)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def forward(self, 
                file_path: str, 
                new_content: str,
                start_anchor: str,
                end_anchor: str,
                create_if_missing: bool = False,
                backup: bool = True,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Insert content between two anchor points in a file.
        
        Args:
            file_path: Path to the target file
            new_content: Content to insert between the anchors
            start_anchor: Text marking the beginning of the insertion point
            end_anchor: Text marking the end of the insertion point
            create_if_missing: Create the file with anchors if it doesn't exist
            backup: Create a backup of the original file before modifying
            verbose: Print detailed information about the operation
            
        Returns:
            Dictionary with operation results including:
            - success: Whether the operation was successful
            - file_path: Path to the modified file
            - backup_path: Path to the backup file (if created)
            - message: Description of the operation result
        """
        file_path = abspath(file_path)
        
        # Check if file exists
        if not os.path.exists(file_path):
            if not create_if_missing:
                if verbose:
                    print(f"File not found: {file_path}", color="red")
                return {
                    "success": False,
                    "file_path": file_path,
                    "message": f"File not found and create_if_missing is False"
                }
            else:
                # Create new file with anchors and content
                content = f"{start_anchor}\n{new_content}\n{end_anchor}"
                put_text(file_path, content)
                if verbose:
                    print(f"Created new file with content: {file_path}", color="green")
                return {
                    "success": True,
                    "file_path": file_path,
                    "message": "Created new file with anchors and content"
                }
        
        # Read the original content
        original_content = get_text(file_path)
        if original_content is None:
            if verbose:
                print(f"Could not read file: {file_path}", color="red")
            return {
                "success": False,
                "file_path": file_path,
                "message": "Could not read file content"
            }
        
        # Create backup if requested
        backup_path = None
        if backup:
            backup_path = f"{file_path}.bak"
            put_text(backup_path, original_content)
            if verbose:
                print(f"Created backup: {backup_path}", color="blue")
        
        # Find the anchor points
        pattern = re.escape(start_anchor) + r"(.*?)" + re.escape(end_anchor)
        match = re.search(pattern, original_content, re.DOTALL)
        
        if match:
            # Replace content between anchors
            new_file_content = original_content.replace(
                f"{start_anchor}{match.group(1)}{end_anchor}",
                f"{start_anchor}\n{new_content}\n{end_anchor}"
            )
            if verbose:
                print(f"Found anchors and replacing content", color="green")
        else:
            # Anchors not found - append to end of file
            if verbose:
                print(f"Anchors not found in file, appending to end", color="yellow")
            new_file_content = f"{original_content}\n\n{start_anchor}\n{new_content}\n{end_anchor}"
        
        # Write the modified content
        put_text(file_path, new_file_content)
        
        if verbose:
            print(f"Successfully inserted content in: {file_path}", color="green")
        
        return {
            "success": True,
            "file_path": file_path,
            "backup_path": backup_path,
            "message": "Successfully inserted content between anchors"
        }
    