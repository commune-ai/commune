 # start of file
import os
import pandas as pd
from typing import List, Dict, Union, Optional, Any
import commune as c
import subprocess
import json

import pandas as pd
import subprocess
import json
import os
from typing import Optional, List, Dict, Any
from datetime import datetime


class Docker:
    """
    A module for interacting with Docker.
    """
    default_shm_size = '100g'
    default_network = 'host'

    def __init__(self):
        pass

    def rmi(self, *images): 
        for image in images:
            try:
                c.cmd(f'docker rmi {image}')
            except Exception as e:
                c.print(f"Error removing image {image}: {e}", color='red')
        

    def build(self,
              path: Optional[str] = './',
              tag: Optional[str] = None,
              sudo: bool = False,
              verbose: bool = True,
              no_cache: bool = False,
              env: Dict[str, str] = {}) -> Dict[str, Any]:
        """
        Build a Docker image from a Dockerfile.

        Args:
            path (Optional[str]): Path to the Dockerfile. Defaults to None.
            tag (Optional[str]): Tag for the image. Defaults to None.
            sudo (bool): Use sudo. Defaults to False.
            verbose (bool): Enable verbose output. Defaults to True.
            no_cache (bool): Disable cache during build. Defaults to False.
            env (Dict[str, str]): Environment variables. Defaults to {}.

        Returns:
            Dict[str, Any]: A dictionary containing the status, tag, and result of the build.
        """
        path = os.path.abspath(path)
        tag = tag or path.split('/')[-1]
        cmd = f'docker build -t {tag} .'
        if no_cache:
            cmd += ' --no-cache'
        
        return c.cmd(cmd,  cwd=path, verbose=True)

    def run(self,
            image  = 'commune',
            cmd: Optional[str] = None,
            name: Optional[str] = None,
            volumes: Dict[str, str] = None,
            gpus: Union[List[int], str, bool] = None,
            shm_size: str = '100g',
            entrypoint = 'tail -f /dev/null',
            sudo: bool = False,
            build: bool = True,
            ports: Optional[Dict[str, int]] = None,
            net: str = 'host',
            daemon: bool = True,
            cwd: Optional[str] = None,
            env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Run a Docker container with advanced configuration options.

        Args:
            path (str): Path to Dockerfile or image name.
            cmd (Optional[str]): Command to run in container.
            volumes (Optional[Union[List[str], Dict[str, str], str]]): Volume mappings.
            name (Optional[str]): Container name.
            gpus (Union[List[int], str, bool]): GPU configuration.
            shm_size (str): Shared memory size.
            sudo (bool): Use sudo.
            build (bool): Build image before running.
            ports (Optional[Dict[str, int]]): Port mappings.
            net (str): Network mode.
            daemon (bool): Run in daemon mode.
            cwd (Optional[str]): Working directory.
            env (Optional[Dict[str, str]]): Environment variables.

        Returns:
            Dict[str, Any]: A dictionary containing the command and working directory.
        """
        name = name or image
        self.kill(name)
        docker_cmd = ['docker', 'run']
        docker_cmd.extend(['--net', net])
        # Handle GPU configuration
        if gpus != None:
            if isinstance(gpus, list):
                docker_cmd.append(f'--gpus "device={",".join(map(str, gpus))}"')
            elif isinstance(gpus, str):
                docker_cmd.append(f'--gpus "{gpus}"')
            elif gpus in [True, 'all']:
                docker_cmd.append(f'--gpus all')
        
        # Configure shared memory
        if shm_size != None:
            docker_cmd.extend(['--shm-size', shm_size])

        # Handle port mappings
        if ports:
            if isinstance(ports, list):
                ports = {port: port for port in ports}
            assert isinstance(ports, dict) and \
                 all([isinstance(k, int) for k in ports.keys()]) and all([isinstance(v, int) for v in ports.values()]), f'ports should be a dict of int to int, got {type(ports)}'
            for host_port, container_port in ports.items():
                docker_cmd.extend(['-p', f'{host_port}:{container_port}'])
        # Handle volumes mappings
        if volumes:
            assert isinstance(volumes, dict)
            volumes = [f'{k}:{v}' for k, v in volumes.items()]
            for k,v in volumes.items():
                docker_cmd.extend(['-v', f'{k}:{v}'])
        # Handle environment variables
        if env:
            for key, value in env.items():
                docker_cmd.extend(['-e', f'{key}={value}'])

        # Set container name
        if name:
            docker_cmd.extend(['--name', name])

        # Run in daemon mode

        if cmd:
            docker_cmd.extend(['--command', f'bash -c "{cmd}"'])
        else:
            if daemon:
                docker_cmd.append('-d')

        


        # Add image name
        docker_cmd.append(image)

        command_str = ' '.join(docker_cmd)
        print(f'Running command: {command_str}')
        return c.cmd(command_str, verbose=True)
     enter(self, contianer): 
        cmd = f'docker exec -it {contianer} bash'
        os.system(cmd)


    def exists(self, name: str) -> bool:
        """
        Check if a container exists.

        Args:
            name (str): The name of the container.

        Returns:
            bool: True if the container exists, False otherwise.
        """
        return name in self.ps()
        
    def kill(self, name: str, sudo: bool = False, verbose: bool = True, prune: bool = False) -> Dict[str, str]:
        """
        Kill and remove a container.

        Args:
            name (str): The name of the container.
            sudo (bool): Use sudo.
            verbose (bool): Enable verbose output.
            prune (bool): Prune unused Docker resources.

        Returns:
            Dict[str, str]: A dictionary containing the status and name of the container.
        """
        try:
            c.cmd(f'docker kill {name}', sudo=sudo, verbose=verbose)
            c.cmd(f'docker rm {name}', sudo=sudo, verbose=verbose)
            if prune:
                self.prune()
            return {'status': 'killed', 'name': name}
        except Exception as e:
            return {'status': 'error', 'name': name, 'error': str(e)}

    def kill_all(self, sudo: bool = False, verbose: bool = True) -> Dict[str, str]:
        """
        Kill all running containers.

        Args:
            sudo (bool): Use sudo.
            verbose (bool): Enable verbose output.

        Returns:
            Dict[str, str]: A dictionary indicating the status of the operation.
        """
        try:
            for container in self.ps():
                self.kill(container, sudo=sudo, verbose=verbose)
            return {'status': 'all_containers_killed'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


    def image_exists(self, name: str) -> bool:
        """
        Check if a Docker image exists.

        Args:
            name (str): The name of the image.

        Returns:
            bool: True if the image exists, False otherwise.
        """
        return name in self.images(names_only=True)

    def images(self, names_only: bool = False) -> Union[pd.DataFrame, Any]:
        """
        List all Docker images.

        Args:
            to_records (bool): Convert to records.

        Returns:
            Union[pd.DataFrame, Any]: A DataFrame or records of Docker images.
        """
        text = c.cmd('docker images')
        rows = []
        for i, line in enumerate(text.split('\n')):
            if not line.strip():
                continue
            if i == 0:
                cols = [col.strip().lower().replace(' ', '_') for col in line.split('  ') if col]
            else:
                rows.append([v.strip() for v in line.split('  ') if v])

        if names_only:
            return [row[0] for row in rows]
        else:
            return  pd.DataFrame(rows, columns=cols)

    def logs(self,
             name: str,
             sudo: bool = False,
             follow: bool = False,
             verbose: bool = False,
             tail: int = 100,
             since: Optional[str] = None) -> str:
        """
        Get container logs with advanced options.

        Args:
            name (str): The name of the container.
            sudo (bool): Use sudo.
            follow (bool): Follow the logs.
            verbose (bool): Enable verbose output.
            tail (int): Number of lines to tail.
            since (Optional[str]): Show logs since timestamp.

        Returns:
            str: The container logs.
        """
        cmd = ['docker', 'logs']

        if tail:
            cmd.extend(['--tail', str(tail)])
        if since:
            cmd.extend(['--since', since])

        cmd.append(name)
        try:
            return c.cmd(' '.join(cmd), verbose=verbose)
        except Exception as e:
            return f"Error fetching logs: {e}"

    def prune(self, all: bool = False) -> str:
        """
        Prune Docker resources.

        Args:
            all (bool): Prune all unused resources.

        Returns:
            str: The result of the prune command.
        """
        cmd = 'docker system prune -f' if all else 'docker container prune -f'
        try:
            return c.cmd(cmd)
        except Exception as e:
            return f"Error pruning: {e}"

    def get_path(self, path: str) -> str:
        """
        Get the path to a Docker-related file.

        Args:
            path (str): The path to the file.

        Returns:
            str: The full path to the file.
        """
        return os.path.expanduser(f'~/.commune/docker/{path}')

    def stats(self, max_age=60, update=False) -> pd.DataFrame:
        """
        Get container resource usage statistics.

        Args:
            max_age (int): Maximum age of cached data in seconds.
            update (bool): Force update of data.

        Returns:
            pd.DataFrame: A DataFrame containing the container statistics.
        """
        path = self.get_path(f'container_stats.json')
        stats = c.get(path, [], max_age=max_age, update=update)
        if len(stats) == 0:
            cmd = f'docker stats --no-stream'
            output = c.cmd(cmd, verbose=False)
            lines = output.split('\n')
            headers = lines[0].split('  ')
            lines = [line.split('   ') for line in lines[1:] if line.strip()]
            lines = [[col.strip().replace(' ', '') for col in line if col.strip()] for line in lines]
            headers = [header.strip().replace(' %', '') for header in headers if header.strip()]
            data = pd.DataFrame(lines, columns=headers)
            stats = []
            for k, v in data.iterrows():
                row = {header: v[header] for header in headers}
                if 'MEM USAGE / LIMIT' in row:
                    mem_usage, mem_limit = row.pop('MEM USAGE / LIMIT').split('/')
                    row['MEM_USAGE'] = mem_usage
                    row['MEM_LIMIT'] = mem_limit
                c.print(row)
                row['ID'] = row.pop('CONTAINER ID')

                for prefix in ['NET', 'BLOCK']:
                    if f'{prefix} I/O' in row:
                        net_in, net_out = row.pop(f'{prefix} I/O').split('/')
                        row[f'{prefix}_IN'] = net_in
                        row[f'{prefix}_OUT'] = net_out
                
                row = {_k.lower(): _v for _k, _v in row.items()}
                stats.append(row)
                c.put(path, stats)
            
        return c.df(stats)

    def ps(self) -> List[str]:
        """
        List all running Docker containers.

        Returns:
            List[str]: A list of container names.
        """
        try:
            text = c.cmd('docker ps')
            ps = []
            for i, line in enumerate(text.split('\n')):
                if not line.strip():
                    continue
                if i > 0:
                    parts = line.split()
                    if len(parts) > 0:  # Check if there are any parts in the line
                        ps.append(parts[-1])
            return ps
        except Exception as e:
            c.print(f"Error listing containers: {e}", color='red')
            return []

    def exec(self, name: str, cmd: str, *extra_cmd) -> str:
        """
        Execute a command in a running Docker container.
        """
        if len(extra_cmd) > 0:
            cmd = ' '.join([cmd] + list(extra_cmd))
        return c.cmd(f'docker exec {name} bash -c "{cmd}"')


    def containers(self) -> List[str]:
        """
        List all Docker containers.
        """
        try:
            text = c.cmd('docker ps')
            ps = []
            for i, line in enumerate(text.split('\n')):
                if not line.strip():
                    continue
                if i > 0:
                    parts = line.split()
                    if len(parts) > 0:  # Check if there are any parts in the line
                        ps.append(parts[-1])
            return ps
        except Exception as e:
            c.print(f"Error listing containers: {e}", color='red')
            return []

    def sync(self):
        """
        Sync container statistics.
        """
        self.stats(update=1)

    # PM2-like methods for container management
    def start(self, name: str, image: str, **kwargs) -> Dict[str, Any]:
        """
        Start a container (PM2-like interface).
        """
        if self.exists(name):
            return self.restart(name)
        
        return self.run(image=image, name=name, **kwargs)

    def stop(self, name: str) -> Dict[str, str]:
        """
        Stop a container without removing it (PM2-like interface).
        """
        try:
            c.cmd(f'docker stop {name}', verbose=False)
            return {'status': 'stopped', 'name': name}
        except Exception as e:
            return {'status': 'error', 'name': name, 'error': str(e)}

    def restart(self, name: str) -> Dict[str, str]:
        """
        Restart a container (PM2-like interface).
        """
        try:
            c.cmd(f'docker restart {name}', verbose=False)
            return {'status': 'restarted', 'name': name}
        except Exception as e:
            return {'status': 'error', 'name': name, 'error': str(e)}