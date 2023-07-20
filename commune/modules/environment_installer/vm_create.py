
import subprocess
import os
import getpass

class VMManager:
    def init(self):
        self.install_libvirt()

    def install_libvirt(self):
        # Install libvirt packages
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'libvirt-dev'])

    def create_vm(self, vm_name, disk_path, memory_size, vcpu_count):
        # Define the virt-install command to create the VM
        command = [
            'virt-install',
            '--name', vm_name,
            '--memory', str(memory_size),
            '--vcpus', str(vcpu_count),
            '--disk', 'path={}'.format(disk_path),
            '--os-variant', 'ubuntu22.04',
            '--graphics', 'none',
            '--console', 'pty,target_type=serial',
            '--import'
        ]

        # Execute the virt-install command to create the VM
        subprocess.run(command)

def create_folder():
    folder_name = "vm_disks"
    home_dir = os.path.expanduser("~")
    folder_path = os.path.join(home_dir, folder_name)

    # Check if the folder already exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created successfully at: {folder_path}")
    else:
        print(f"Folder '{folder_name}' already exists at: {folder_path}")

def create_empty_disk_image(disk_path, size_gb):
    command = ['qemu-img', 'create', '-f', 'qcow2', disk_path, str(size_gb) + 'G']
    subprocess.run(command)

def run_basic_tests():
    vm_manager = VMManager()

    # Test 1: Create a VM
    vm_name = "vm1"
    username = getpass.getuser()
    disk_path = "/home/" + username + "/vm_disks/vm1.qcow2"
    if not os.path.exists(disk_path):
        create_empty_disk_image(disk_path, 20)  # Create a 20GB disk image
    memory_size = 2048
    vcpu_count = 2

    vm_manager.create_vm(vm_name, disk_path, memory_size, vcpu_count)
    print("Test 1: VM created successfully.")

    # Test 2: Create another VM
    vm_name = "vm2"
    disk_path = "/home/" + username + "/vm_disks/vm2.qcow2"
    if not os.path.exists(disk_path):
        create_empty_disk_image(disk_path, 20)  # Create a 20GB disk image
    memory_size = 4096
    vcpu_count = 4

    vm_manager.create_vm(vm_name, disk_path, memory_size, vcpu_count)
    print("Test 2: VM created successfully.")

if __name__ == "__main__":
    create_folder()
    run_basic_tests()