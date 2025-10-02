
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

    
    def pipe(
        self,
        commands: List[str],
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a pipeline of commands, passing output from one to the next.
        
        Args:
            commands: List of commands to execute in sequence
            verbose: Whether to print command and output information
            **kwargs: Additional arguments to pass to forward()
            
        Returns:
            Dictionary with the result of the final command
        """
        if not commands:
            return {"success": False, "error": "no_commands", "message": "No commands provided"}
        
        if verbose:
            c.print(f"Executing pipeline of {len(commands)} commands", color="cyan")
        
        result = None
        for i, cmd in enumerate(commands):
            if verbose:
                c.print(f"Step {i+1}/{len(commands)}: {cmd}", color="blue")
                
            if result is not None and result.get("stdout"):
                # Use previous command's output as input
                if kwargs.get("shell", self.shell):
                    cmd = f"echo '{result['stdout']}' | {cmd}"
                else:
                    # For non-shell execution, we need a different approach
                    result = self.forward(
                        f"bash -c \"echo '{result['stdout']}' | {cmd}\"",
                        shell=True,
                        verbose=verbose,
                        **kwargs
                    )
                    continue
                    
            result = self.forward(cmd, verbose=verbose, **kwargs)
            
            if not result["success"]:
                if verbose:
                    c.print(f"Pipeline failed at step {i+1}", color="red")
                break
                
        return result
