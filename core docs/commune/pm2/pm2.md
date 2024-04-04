## Readme

#### Description
This file contains a class PM2 to interact with and manage the pm2 Node.js process manager. It includes methods for creating, listing, restarting, and stopping processes. Certain commands are extended to use 'starts with' phrase finding. The results of certain commands will be printed in color for emphasis.  

#### Methods:

- **restart:**: Restarts a PM2 process given by name. 

- **restart_prefix:**: Restarts the PM2 processes whose names start with the given prefix.

- **kill:**: Kills a PM2 process specified by a name. 

- **status:**: Shows the status of PM2 processes. 

- **logs_path_map:**: Gives the path of log files for a given PM2 process.

- **rm_logs:**:  Removes the log files for a PM2 process.

- **logs:**: Fetches and prints the logs for a given PM2 process.

- **kill_many:**: Kills several PM2 processes concurrently and waits for all of them to complete.

- **kill_all:**: Kills all PM2 processes. 

- **list:**: Lists all the PM2 processes.

- **exists:**: Checks if a PM2 process with a given name exists.

- **start:**: Starts a PM2 process. 

- **launch:**: Creates and starts a PM2 process.

#### Usage:
- Import the class: `from filename import PM2`
- Create an instance: `pm2 = PM2()`
- Call a method: `pm2.kill('process_name')`

Note that this code uses the 'commune' package (aliased as `c`), and assumes that the `pm2` command line tool is installed on the system and available in the system's PATH. It also assumes that PM2 process logs are stored in '~/.pm2'. You need to have `commune` installed in your system and `Node.js` installed with `pm2` module to run the functions in the class.

The class methods `kill`, `restart`, and `restart_prefix` accept an optional verbosity flag, which defaults to False, and a prefix_match flag, which defaults to True. When the verbose flag is True, additional status messages will be printed for the user to see. When the prefix_match is True, methods will search for processes that start with the given name pattern. 

Code also uses some color highlights like 'cyan', 'red', 'green' to give colorful outputs in the terminal.

This is a general-purpose PM2 manager and can be used in any code requiring process management with PM2.