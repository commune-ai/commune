
import commune as c
import subprocess
import shlex
import os
from typing import List, Dict, Union, Optional, Any

class Cmd:
    """
    Command-line execution tool for running shell commands with various options.
    
    This tool provides a clean interface for executing shell commands with options
    for capturing output, handling errors, and processing results.
    """
    
    def __init__(self, cwd: str = None, shell: bool = False, env: Dict[str, str] = None):
        """
        Initialize the Cmd tool.
        
        Args:
            cwd: Current working directory for command execution
            shell: Whether to use shell execution (can be a security risk)
            env: Environment variables to set for command execution
        """
        self.cwd = cwd
        self.shell = shell
        self.env = env
        
    def forward(
        self,
        cmd: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        Execute a shell command and return the results.
        """
        # Use instance defaults if not specified
        result = os.system(cmd, **kwargs)
        return result