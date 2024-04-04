# SSH Module

This module allows running commands on a remote server using SSH.

## Codes & Functions

The SSH Module contains the following functions:

**Initialization**:  

The SSH class inherits from commune's("c") `Module` class. It contains one class variable `host_data_path` which is a path to SSH data.

**Methods**:  

1. `call`: This method runs a command on a remote server using SSH. It allows you to specify the host, port, username, password, and the command to be executed.

2. `serve`: This method invokes the `call` method.

3. `add_host`: This method adds a host to the list of hosts.

4. `save_hosts`: This method saves the hosts data to a JSON file.

5. `load_hosts`: This method loads the hosts data from a JSON file.

6. `switch_hosts`: This method switches the host data to another path.

7. `rm_host`: This method removes a specific host from the list of hosts.

8. `hosts`: This method retrieves the list of hosts.

9. `host`: This method retrieves the data of a specific host.

10. `host_exists`: This method checks if a host exists in the list of hosts.

11. `install`: This method installs the necessary dependencies for the module.

12. `test`: This method tests the SSH module.

13. `pool`: This method runs commands on multiple hosts simultaneously.

## How to use

First, import the module using `import commune as c`.

Next, instantiate a new object of the class SSH, for example: `ssh_obj = SSH()`.

You can use any of the mentioned methods on this object to establish and operate SSH connections. 

For example, `ssh_obj.call('ls')` will execute the 'ls' command on the previously mentioned host.
