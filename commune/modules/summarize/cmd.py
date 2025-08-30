
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
        command: Union[str, List[str]],
        capture_output: bool = True,
        text: bool = True,
        check: bool = False,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        shell: Optional[bool] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a shell command and return the results.
        
        Args:
            command: Command to execute (string or list of arguments)
            capture_output: Whether to capture stdout/stderr
            text: Whether to return strings instead of bytes
            check: Whether to raise an exception on non-zero exit code
            timeout: Maximum time to wait for command completion (in seconds)
            cwd: Working directory for command execution (overrides instance setting)
            env: Environment variables (overrides instance setting)
            shell: Whether to use shell execution (overrides instance setting)
            verbose: Whether to print command and output information
            
        Returns:
            Dictionary containing:
            - success: Whether the command executed successfully
            - returncode: Exit code of the command
            - stdout: Standard output (if captured)
            - stderr: Standard error (if captured)
            - command: The command that was executed
        """
        # Use instance defaults if not specified
        cwd = cwd if cwd is not None else self.cwd
        env = env if env is not None else self.env
        shell = shell if shell is not None else self.shell
        
        # Process the command
        if isinstance(command, str) and not shell:
            command = shlex.split(command)
        
        if verbose:
            c.print(f"Executing: {command}", color="cyan")
        
        try:
            # Execute the command
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=text,
                check=check,
                timeout=timeout,
                cwd=cwd,
                env=env,
                shell=shell
            )
            
            # Prepare the result dictionary
            output = {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "command": command
            }
            
            # Add stdout/stderr if captured
            if capture_output:
                output["stdout"] = result.stdout
                output["stderr"] = result.stderr
                
                if verbose:
                    if result.stdout:
                        c.print("STDOUT:", color="green")
                        c.print(result.stdout)
                    if result.stderr:
                        c.print("STDERR:", color="red")
                        c.print(result.stderr)
            
            if verbose:
                status = "Success" if output["success"] else f"Failed (code: {result.returncode})"
                c.print(f"Command execution: {status}", color="green" if output["success"] else "red")
                
            return output
            
        except subprocess.TimeoutExpired as e:
            if verbose:
                c.print(f"Command timed out after {timeout} seconds", color="red")
            return {
                "success": False,
                "error": "timeout",
                "command": command,
                "timeout": timeout,
                "message": str(e)
            }
            
        except subprocess.SubprocessError as e:
            if verbose:
                c.print(f"Command execution error: {e}", color="red")
            return {
                "success": False,
                "error": "subprocess_error",
                "command": command,
                "message": str(e)
            }
            
        except Exception as e:
            if verbose:
                c.print(f"Unexpected error: {e}", color="red")
            return {
                "success": False,
                "error": "unexpected",
                "command": command,
                "message": str(e)
            }

        return {}
    
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
