import os
import subprocess
import getpass

import os
import getpass
import commune as c

class Ansible(c.Module):
    def __init__(self, inventory_file=c.libpath +'/data/inventory.ini'):
        self.inventory_file = inventory_file
        assert os.path.exists(self.inventory_file), f"Inventory file not found: {self.inventory_file}"

        self.host_counter = self.get_last_host_number() + 1

    def get_last_host_number(self):
        if os.path.exists(self.inventory_file):
            with open(self.inventory_file, 'r') as file:
                lines = [line.strip() for line in file.readlines() if line.strip()]
                if lines:
                    last_host = lines[-1].split()[0].replace('host', '')
                    return int(last_host)
        return 0

    def prompt_user(self):
        ip = input("Remote server IP: ").strip()
        port = input("Remote server port: ").strip()
        username = input("Remote server username: ").strip()
        print("Remote server password: ", end="")
        password = getpass.getpass()
        return ip, port, username, password

    def update_inventory(self, ip, port, username, password):
        host_line = f"host{self.host_counter} ansible_host={ip} ansible_user={username} ansible_ssh_pass={password} ansible_port={port}"

        with open(self.inventory_file, 'a') as file:
            if self.host_counter == 1:  # or another condition to decide when to add [all]
                file.write("[all]\n")
            file.write(host_line + "\n")
        print(f"Added host{self.host_counter} to {self.inventory_file}")

        self.host_counter += 1

    def ping(self):
        return c.cmd('ansible all -i '+ self.inventory_file +' -m ping -u root')
    def cmd(self, command, pwd=None):
        # Disable Host Key Checking
        os.environ['ANSIBLE_HOST_KEY_CHECKING'] = 'False'

        # # Get SSH Password
        # if pwd is None:
        #     pwd = getpass.getpass()

        try:
            output = subprocess.check_output(
                [
                    'ansible',
                    'all',
                    '-i', self.inventory_file,
                    '-m', 'shell',
                    '-a', command,
                    '-u', 'root',
                    # '--ask-pass',
                    # '--extra-vars', f'ansible_ssh_pass={pwd}'
                ],
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            print("Output:")
            print(output)

        except subprocess.CalledProcessError as error:
            print("Error:")
            print(error.output)
