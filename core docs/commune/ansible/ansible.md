# Ansible Module for Commune

The provided Python script presents an abstracted module for Ansible, a renowned IT automation tool. This Ansible module simplifies interaction with remote servers, facilitating user prompts for connection details, and automated inventory updating with Ansible ping capability.

## Overview

This Ansible module enhances the Commune software, easing the incorporation of new hosts in the inventory. This is achieved by prompting the user to input the specifics of the remote server, such as IP address, port, username and password. After acquiring the details, the module automatically updates the Ansible inventory file and increments a counter to keep track of the number of hosts.

## Requirements

- **Python**: The script is written using Python, hence Python installation with version 3.6 or later is necessary.
- **Ansible**: As this module is associated with Ansible, ensure Ansible is installed and properly configured.
- **Commune**: The module is designed as an expansion of Commune software, thus it's required to be installed too.

## Usage

### 1. Instantiate Ansible Module

You can create an instance of the Ansible module by initializing it with an inventory file.

    my_ansible = Ansible('/path/to/my/inventory.ini')

### 2. Prompt User

To gather the specifics of the remote server from the user, simply invoke the `prompt_user()` method.

    server_details = my_ansible.prompt_user()

### 3. Update Inventory

To add the new host to the Ansible inventory, utilize `update_inventory()`.

    my_ansible.update_inventory(*server_details)

### 4. Ping Hosts

To check the connectivity to all hosts in the Ansible inventory, use the `ping()` method.

    my_ansible.ping()

### 5. Send Command to Hosts

To send a shell command to all hosts in the Ansible inventory, apply the `cmd()` method.

    my_ansible.cmd("date")

## Troubleshooting

- **Ansible Configuration**: Ensure that Ansible is installed and properly configured on your system.
- **Commune Installation**: Validate the successful installation and configuration of the Commune software.
- **Host Connectivity**: Certify the remote server provided by the user is accessible from your system.

## Support

For any queries, issues or support you can contact via the options provided in the linked contact page.

## License

This Ansible module for Commune falls under the criteria of the MIT license.
