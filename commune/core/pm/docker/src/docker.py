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
        tag = tag or path.split('/')[-2]
        cmd = f'docker build -t {tag} .'
        if no_cache:
            cmd += ' --no-cache'
        return c.cmd(cmd,  cwd=path)

    def run(self,
            image  = 'commune',
            cmd: Optional[str] = None,
            *extra_cmd,

            name: Optional[str] = 'commune',

            vol: Dict[str, str] = None,
            gpus: Union[List[int], str, bool] = False,
            shm_size: str = '100g',
            entrypoint = 'tail -f /dev/null',
            sudo: bool = False,
            build: bool = True,
            ports: Optional[Dict[str, int]] = None,
            net: str = 'host',
            daemon: bool = False,
            cwd: Optional[str] = None,
            env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Run a Docker container with advanced configuration options.

        Args:
            path (str): Path to Dockerfile or image name.
            cmd (Optional[str]): Command to run in container.
            vol (Optional[Union[List[str], Dict[str, str], str]]): Volume mappings.
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
        cmd = cmd 
        if len(extra_cmd) > 0:
            cmd = ' '.join([cmd] + list(extra_cmd))
        dcmd = ['docker', 'run']
        dcmd.extend(['--net', net])
        # Handle GPU configuration
        if isinstance(gpus, list):
            dcmd.append(f'--gpus "device={",".join(map(str, gpus))}"')
        elif isinstance(gpus, str):
            dcmd.append(f'--gpus "{gpus}"')
        elif gpus is True:
            dcmd.append(f'--gpus all')
        # Configure shared memory
        if shm_size:
            dcmd.extend(['--shm-size', shm_size])
        # Handle port mappings
        if ports:
            if isinstance(ports, list):
                ports = {port: port for port in ports}
            for host_port, container_port in ports.items():
                dcmd.extend(['-p', f'{host_port}:{container_port}'])
            
        # Handle volume mappings
        if vol:
            assert isinstance(vol, dict)
            vol = [f'{k}:{v}' for k, v in vol.items()]
            for volume in vol:
                dcmd.extend(['-v', volume])
        # Handle environment variables
        if env:
            for key, value in env.items():
                dcmd.extend(['-e', f'{key}={value}'])

        # Set container name
        if name:
            dcmd.extend(['--name', name])

        # Run in daemon mode
        if daemon:
            dcmd.append('-d')


        if cmd:
            dcmd.extend(['--entrypoint', f'bash -c "{cmd}"'])
        


        # Add image name
        dcmd.append(image)

        command_str = ' '.join(dcmd)
        print(f'Running command: {command_str}')
        return c.cmd(command_str, verbose=True)


    def enter(self, contianer): 
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

    def images(self, df: bool = True) -> Union[pd.DataFrame, Any]:
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
            print(line)
            if not line.strip():
                continue
            if i == 0:
                cols = [col.strip().lower().replace(' ', '_') for col in line.split('  ') if col]
            else:
                rows.append([v.strip() for v in line.split('  ') if v])

        results = pd.DataFrame(rows, columns=cols)
        if not df:
            return results['repository'].tolist()
        else:
            return results

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

        Args:
            name (str): The name of the container.
            cmd (str): The command to execute.
            *extra_cmd: Additional command arguments.

        Returns:
            str: The output of the command.
        """
        if len(extra_cmd) > 0:
            cmd = ' '.join([cmd] + list(extra_cmd))
        
        return c.cmd(f'docker exec {name} bash -c "{cmd}"')

    def cstats(self, max_age=10, update=False, cache_dir="./docker_stats") -> pd.DataFrame:
        """
        Get resource usage statistics for all containers.

        Args:
            max_age (int): Maximum age of cached data in seconds
            update (bool): Force update of data
            cache_dir (str): Directory to store cached data

        Returns:
            pd.DataFrame: A DataFrame containing statistics for all containers
        """
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "all_containers.json")
        
        # Check if cache exists and is recent enough
        should_update = update
        if not should_update and os.path.exists(cache_file):
            file_age = datetime.now().timestamp() - os.path.getmtime(cache_file)
            should_update = file_age > max_age
        
        if should_update or not os.path.exists(cache_file):
            # Run docker stats command
            cmd = 'docker stats --no-stream'
            try:
                output = subprocess.check_output(cmd, shell=True, text=True)
            except subprocess.CalledProcessError:
                print("Error running docker stats command")
                return pd.DataFrame()
            
            # Parse the output
            lines = output.strip().split('\n')
            if len(lines) <= 1:
                print("No containers running")
                return pd.DataFrame()
            
            # Process headers
            headers = [h.strip() for h in lines[0].split('  ') if h.strip()]
            cleaned_headers = []
            header_indices = []
            
            # Find the position of each header in the line
            current_pos = 0
            for header in headers:
                pos = lines[0].find(header, current_pos)
                if pos != -1:
                    header_indices.append(pos)
                    cleaned_headers.append(header)
                    current_pos = pos + len(header)
            
            # Process data rows
            stats = []
            for line in lines[1:]:
                if not line.strip():
                    continue
                    
                # Extract values based on header positions
                values = []
                for i in range(len(header_indices)):
                    start = header_indices[i]
                    end = header_indices[i+1] if i+1 < len(header_indices) else len(line)
                    values.append(line[start:end].strip())
                
                # Create a dictionary for this row
                row = dict(zip(cleaned_headers, values))
                
                # Process special columns
                if 'MEM USAGE / LIMIT' in row:
                    mem_usage, mem_limit = row.pop('MEM USAGE / LIMIT').split('/')
                    row['MEM_USAGE'] = mem_usage.strip()
                    row['MEM_LIMIT'] = mem_limit.strip()
                
                for prefix in ['NET', 'BLOCK']:
                    if f'{prefix} I/O' in row:
                        io_in, io_out = row.pop(f'{prefix} I/O').split('/')
                        row[f'{prefix}_IN'] = io_in.strip()
                        row[f'{prefix}_OUT'] = io_out.strip()
                
                # Rename ID column
                if 'CONTAINER ID' in row:
                    row['ID'] = row.pop('CONTAINER ID')
                
                # Convert keys to lowercase
                row = {k.lower(): v for k, v in row.items()}
                stats.append(row)
            
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(stats, f)
        else:
            # Load from cache
            with open(cache_file, 'r') as f:
                stats = json.load(f)
        
        # Convert to DataFrame
        return pd.DataFrame(stats)

    def sync(self):
        """
        Sync container statistics.
        """
        self.stats(update=1)

    # PM2-like methods for container management
    def start(self, name: str, image: str, **kwargs) -> Dict[str, Any]:
        """
        Start a container (PM2-like interface).

        Args:
            name (str): Name for the container.
            image (str): Docker image to use.
            **kwargs: Additional arguments for the run method.

        Returns:
            Dict[str, Any]: Result of the operation.
        """
        if self.exists(name):
            return self.restart(name)
        
        return self.run(image=image, name=name, **kwargs)

    def stop(self, name: str) -> Dict[str, str]:
        """
        Stop a container without removing it (PM2-like interface).

        Args:
            name (str): The name of the container.

        Returns:
            Dict[str, str]: Result of the operation.
        """
        try:
            c.cmd(f'docker stop {name}', verbose=False)
            return {'status': 'stopped', 'name': name}
        except Exception as e:
            return {'status': 'error', 'name': name, 'error': str(e)}

    def restart(self, name: str) -> Dict[str, str]:
        """
        Restart a container (PM2-like interface).

        Args:
            name (str): The name of the container.

        Returns:
            Dict[str, str]: Result of the operation.
        """
        try:
            c.cmd(f'docker restart {name}', verbose=False)
            return {'status': 'restarted', 'name': name}
        except Exception as e:
            return {'status': 'error', 'name': name, 'error': str(e)}

    def delete(self, name: str) -> Dict[str, str]:
        """
        Remove a container (PM2-like interface).

        Args:
            name (str): The name of the container.

        Returns:
            Dict[str, str]: Result of the operation.
        """
        return self.kill(name)

    def list(self, all: bool = False) -> pd.DataFrame:
        """
        List containers with detailed information (PM2-like interface).

        Args:
            all (bool): Include stopped containers.

        Returns:
            pd.DataFrame: DataFrame containing container information.
        """
        try:
            cmd = 'docker ps' if not all else 'docker ps -a'
            output = c.cmd(cmd, verbose=False)
            lines = output.split('\n')
            
            if len(lines) <= 1:
                return pd.DataFrame()
                
            # Process headers
            headers = []
            current_pos = 0
            header_line = lines[0]
            
            # Extract header positions
            for i, char in enumerate(header_line):
                if char.isupper() and (i == 0 or header_line[i-1].isspace()):
                    if current_pos < i:
                        header_end = i
                        header_text = header_line[current_pos:header_end].strip()
                        if header_text:
                            headers.append((current_pos, header_text))
                    current_pos = i
            
            # Add the last header
            if current_pos < len(header_line):
                headers.append((current_pos, header_line[current_pos:].strip()))
            
            # Extract header positions for parsing
            header_positions = [pos for pos, _ in headers]
            header_names = [name.lower().replace(' ', '_') for _, name in headers]
            
            # Parse data rows
            data = []
            for line in lines[1:]:
                if not line.strip():
                    continue
                    
                row = {}
                for i in range(len(header_positions)):
                    start = header_positions[i]
                    end = header_positions[i+1] if i+1 < len(header_positions) else len(line)
                    value = line[start:end].strip()
                    row[header_names[i]] = value
                
                data.append(row)
            
            return pd.DataFrame(data)
        except Exception as e:
            c.print(f"Error listing containers: {e}", color='red')
            return pd.DataFrame()

    def monitor(self) -> pd.DataFrame:
        """
        Monitor containers (PM2-like interface).

        Returns:
            pd.DataFrame: DataFrame with container monitoring information.
        """
        return self.cstats(update=True)

    def save(self, config_name: str = 'default') -> Dict[str, Any]:
        """
        Save current container configuration (PM2-like interface).

        Args:
            config_name (str): Name for the configuration.

        Returns:
            Dict[str, Any]: Result of the operation.
        """
        try:
            containers = self.list(all=True)
            if containers.empty:
                return {'status': 'error', 'message': 'No containers to save'}
            
            config_path = self.get_path(f'configs/{config_name}.json')
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Get container details including image, ports, vol, etc.
            container_configs = []
            for _, container in containers.iterrows():
                name = container.get('names', '')
                if not name:
                    continue
                    
                # Get detailed container info
                inspect_cmd = f'docker inspect {name}'
                try:
                    inspect_output = c.cmd(inspect_cmd, verbose=False)
                    container_info = json.loads(inspect_output)[0]
                    
                    config = {
                        'name': name,
                        'image': container_info.get('Config', {}).get('Image', ''),
                        'command': container_info.get('Config', {}).get('Cmd', []),
                        'entrypoint': container_info.get('Config', {}).get('Entrypoint', []),
                        'env': container_info.get('Config', {}).get('Env', []),
                        'ports': container_info.get('HostConfig', {}).get('PortBindings', {}),
                        'vol': container_info.get('HostConfig', {}).get('Binds', []),
                        'network_mode': container_info.get('HostConfig', {}).get('NetworkMode', ''),
                        'restart_policy': container_info.get('HostConfig', {}).get('RestartPolicy', {}),
                        'status': container_info.get('State', {}).get('Status', '')
                    }
                    container_configs.append(config)
                except Exception as e:
                    c.print(f"Error inspecting container {name}: {e}", color='yellow')
                    continue
            
            # Save the configuration
            with open(config_path, 'w') as f:
                json.dump(container_configs, f, indent=2)
                
            return {
                'status': 'success', 
                'message': f'Saved {len(container_configs)} container configurations to {config_path}',
                'path': config_path
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def load(self, config_name: str = 'default') -> Dict[str, Any]:
        """
        Load and apply a saved container configuration (PM2-like interface).

        Args:
            config_name (str): Name of the configuration to load.

        Returns:
            Dict[str, Any]: Result of the operation.
        """
        try:
            config_path = self.get_path(f'configs/{config_name}.json')
            if not os.path.exists(config_path):
                return {'status': 'error', 'message': f'Configuration {config_name} not found'}
            
            with open(config_path, 'r') as f:
                container_configs = json.load(f)
            
            results = []
            for config in container_configs:
                name = config.get('name')
                image = config.get('image')
                
                if not name or not image:
                    results.append({'status': 'error', 'message': 'Missing name or image in config'})
                    continue
                
                # Convert ports format
                ports = {}
                for container_port, host_bindings in config.get('ports', {}).items():
                    if host_bindings and len(host_bindings) > 0:
                        host_port = host_bindings[0].get('HostPort')
                        if host_port:
                            ports[host_port] = container_port.split('/')[0]
                
                # Convert vol format
                vol = {}
                for volume in config.get('vol', []):
                    if ':' in volume:
                        host_path, container_path = volume.split(':', 1)
                        vol[host_path] = container_path
                
                # Convert environment variables
                env = {}
                for env in config.get('env', []):
                    if '=' in env:
                        key, value = env.split('=', 1)
                        env[key] = value
                
                # Start the container
                try:
                    result = self.run(
                        image=image,
                        name=name,
                        cmd=' '.join(config.get('command', [])) if config.get('command') else None,
                        entrypoint=' '.join(config.get('entrypoint', [])) if config.get('entrypoint') else None,
                        vol=vol,
                        ports=ports,
                        env=env,
                        net=config.get('network_mode', 'bridge')
                    )
                    results.append({'name': name, 'status': 'started', 'result': result})
                except Exception as e:
                    results.append({'name': name, 'status': 'error', 'error': str(e)})
            
            return {
                'status': 'success',
                'message': f'Loaded {len(results)} containers from {config_name} configuration',
                'results': results
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def file(self, path: str) -> str:
        """
        Get the content of a Dockerfile.

        Args:
            path (str): Path to the directory containing the Dockerfile.

        Returns:
            str: Content of the Dockerfile.
        """
        dockerfile_path = os.path.join(path, 'Dockerfile')
        if os.path.exists(dockerfile_path):
            with open(dockerfile_path, 'r') as f:
                return f.read()
        return f"Dockerfile not found at {dockerfile_path}"

    def files(self, path: str = '.') -> List[str]:
        """
        Find all Dockerfiles in a directory and its subdirectories.

        Args:
            path (str): Root directory to search.

        Returns:
            List[str]: List of paths to Dockerfiles.
        """
        dockerfiles = []
        for root, _, files in os.walk(path):
            if 'Dockerfile' in files:
                dockerfiles.append(os.path.join(root, 'Dockerfile'))
        return dockerfiles
