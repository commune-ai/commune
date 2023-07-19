import paramiko
import time
import socket
from tqdm import tqdm

myusername = 'endless'
portnumber = '4200'
mypassword = 'endless1234'
rebootwaittime = 600

def execute_command(ssh, command, password=None):
    print(command)
    stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)
    total_size = None
    pbar = None

    if password is not None:
        stdin.write(password + '\n')
        stdin.flush()

    while not stdout.channel.exit_status_ready():
        if stdout.channel.recv_ready():
            output = stdout.channel.recv(1024).decode('utf-8')
            print(output, end='')

        if stderr.channel.recv_stderr_ready():
            error = stderr.channel.recv_stderr(1024).decode('utf-8')
            print(error, end='')

    if pbar is not None:
        pbar.close()

    print('\n')

def connect_ssh(host, port, username, password, timeout=10):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    while True:
        try:
            ssh.connect(host, port=port, username=username, password=password, timeout=timeout)
            print("SSH connection established.")
            break
        except (paramiko.SSHException, socket.timeout, paramiko.ssh_exception.NoValidConnectionsError, TimeoutError) as e:
            print("SSH connection failed. Retrying in 30 seconds...")
            time.sleep(30)

    return ssh


if __name__ == '__main__':

    myusername = input("User Name : ")
    myip = input("IP Address: ")
    portnumber = input('Port : ')
    mypassword = input("Password : ")

    ssh = connect_ssh(myip, int(portnumber), myusername, mypassword)
    execute_command(ssh, 'sudo -S wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin', mypassword)
    ssh.close()

input("Tested!, Press Enter to Exit......")
