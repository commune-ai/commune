# README

This is a Python 3 script that contains a class called `OsModule`. It has multiple helper functions that interact with the operating system and environment to perform various operations such as process management, running shell commands, file/directory management, and getting information about the system such as CPU count, memory usage, and more.

## Helper Functions included in `OsModule`.

1. `check_pid(pid)`: Checks for the existence of a Unix PID.

2. `kill_process(pid)`: Kills a process using a PID.

3. `run_command(command)`: Runs a shell command.

4. `path_exists(path)`: Checks if the given path exists.

5. `ensure_path(path)`: Ensures a directory path exists, otherwise, it will create it.

6. `seed_everything(seed)`: Seeds Python environments for reproducibility.

7. `cpu_count()`: Gets the number of CPUs available in the system.

8. `get_env(key)`: Gets the environment variable with the specified key.

9. `set_env(key, value)`: Sets the environment variable with the specified key to the specified value.

10. `get_cwd()`: Returns the current working directory.

11. `set_cwd(path)`: Changes the current working directory to the supplied path.

12. `get_pid()`: Returns the current process ID (PID).

13. `memory_usage_info(fmt='gb')`: Returns the memory usage info of the current process.

14. `virtual_memory_available()`: Returns the amount of virtual memory available.

15. `virtual_memory_total()`: Returns the total amount of virtual memory.

16. `cpu_type()`: Returns the processor type.

17. `cpu_info()`: Returns CPU count and type.

18. `cpu_usage()`: Returns the CPU usage.

19. `num_gpus()`: Returns the number of GPUs available.

20. `add_rsa_key(b=2048, t='rsa')`: Generates an RSA key pair.

21. `cmd(command,verbose=False,env={},sudo=False)`: Runs a shell command with multiple options like verbose, environment variables, and sudo permissions.

22. `format_data_size(x,fmt='b',prettify=False)`: Formats the size of data.

23. `disk_info(path='/')`: Returns disk usage statistics about the path specified.

24. `mv(path1, path2)`: Moves a file or directory.

25. `cp(path1, path2, refresh=False)`: Copies a file or directory.

26. `cuda_available()`: Checks if CUDA is available.

27. `gpu_info()`: Returns GPU memory information.

28. `gpu_map()`: Returns a map of GPU memory information.

29. `hardware()`: Returns information about the hardware including CPU, memory, disk, and GPU.

30. `virtual_memory_percent()`: Returns the percentage of virtual memory in use.

31. `resolve_device(device=None, verbose=True, find_least_used=True)`: Resolves the device to use based on availability and usage.

32. `find_largest_folder(directory)`: Finds the largest folder in the specified directory.

33. `get_folder_size(folder_path)`: Calculates the total size of all files in the specified folder.
  
34. `getcwd()`: Returns the current working directory.

All the functions have been decorated with `staticmethod` or `classmethod`, which means they can be called on the class itself, without needing to instantiate an object of `OsModule` first.