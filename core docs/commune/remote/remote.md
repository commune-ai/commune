# Remote Module

This is a Python module that provides functionalities to manage commands and operations on remote servers. It uses the Paramiko library for SSH communication. This module also includes managing multiple hosts, setting up search terms for better host management, and interaction with the remote servers.

## Features:
- Run commands on remote server: `Remote.ssh_cmd()`
- Add and remove host details: `Remote.add_host()`, `Remote.rm_host()`
- Save and load host details: `Remote.save_hosts()`, `Remote.load_hosts()`
- Get and manipulate server details: `infors()`, `servers()`, `availability_peers()`
- Check peer availability and track their metadata: `check_peers()`, `add_peers()`, `sync()`
- Utility functions to set search terms, convert host to IP, etc.

## Dependencies:
- commune
- streamlit
- typing
- json
- paramiko

## Setup:
1. Instantiate the `Remote` class.
2. Use `add_host()` to add the details of your remote servers. The details include the hostname or IP, port, username, and password. You can also navigate between hosts using `switch_hosts()`, or remove hosts using `rm_host()`.
3. The `ssh_cmd()` method can be used to execute commands on a remote server.
4. The `servers()` method returns peer server details.
5. Use `serve()` to start serving computations on the remote server.
6. Use `availability_peers()` to get a list of peers with enough memory, and `sync()` to save the metadata of all peers.

## Note:
Ensure that the appropriate permissions are set on the remote server for the desired operations.

For safety, the SSH client is set to automatically add the server's host key which should be changed in the production environment. The key should be added to the known hosts manually.