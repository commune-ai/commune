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

class PM:
    """
    A module for interacting with Docker.
    """

    def __init__(self,  
                path = os.path.expanduser('~/commune'), 
                image = None, # default image name is the folder name
                storage_path='~/.commune/server/pm', 
                **kwargs):
    
        self.path = path
        self.image = (image or path.split('/')[-1].lower()) + ':latest'
        self.store = c.mod('store')(storage_path)
        # if not self.is_docker_daemon_on():
        #     self.start_docker_daemon()

        

    def serve(self, 
                module='api', 
                port=None, 
                image=None, 
                working_dir='/app', 
                daemon=True, d = None, # daemon
                remote=True,
                name = None,
                cwd = None,
                include_storage=True,
                **params):
        image = image or self.image
        daemon = daemon if d is None else d
        module = module or 'module'
        port = port or c.free_port()
        fn = 'server/serve'
        params['remote'] = 0
        params['port'] = port
        params_cmd = self.params2cmd(params)
        cmd = f"c server/serve {module} {params_cmd}"
        if not remote:
            return os.system(cmd)
        if '::' in module:
            name = self.name2process(name or module)
            module = module.split('::')[0]
        name = name or module
        lib_path = c.lib_path
        storage_path = c.storage_path
        docker_lib_path = c.lib_path.replace(c.home_path, '/root')
        docker_storage_path = storage_path.replace(c.home_path, '/root')
        volumes = {
            lib_path: docker_lib_path,
            storage_path: docker_storage_path
        }
        if module not in c.anchor_names:
            mod_path = c.dirpath(module)
            docker_mods_path = mod_path.replace(c.home_path, '/root')
            volumes[mod_path] = docker_mods_path
 
        cwd = cwd or c.dp(module)
        params = {
            'name': name, 
            'image': image,
            'port': port,
            'cmd': cmd,
            'working_dir': working_dir, 
            'daemon': daemon,
            'volumes': volumes,
            'cwd': cwd       
            }
        return self.run(name=name, 
                        image=image, 
                        port=port, 
                        cmd=cmd, 
                        working_dir=working_dir, 
                        daemon=daemon, 
                        volumes=volumes, 
                        cwd=cwd)

    def process2name(self, container):
        return container.replace('__', '::')
    
    def name2process(self, name):
        return name.replace('::', '__')

    def servers(self, search=None, **kwargs):
        servers =  list(map(self.process2name, self.ps()))
        servers = [m for m in servers if m != 'commune']
        if search != None:
            servers = [m for m in servers if search in m]
        servers = sorted(list(set(servers)))
        return servers

    def server_exists(self, name):
        return name in self.servers()

    def params2cmd(self, params: Dict[str, Any]) -> str:
        """
        Convert a dictionary of parameters to a command string.
        
        Args:
            params (Dict[str, Any]): Dictionary of parameters.
            
        Returns:
            str: Command string with parameters formatted as key=value pairs.
        """
        for k, v in params.items():
            if isinstance(v, bool):
                params[k] = '1' if v else '0'
            elif isinstance(v, list):
                params[k] = ','.join(map(str, v))
            elif isinstance(v, dict):
                params[k] = json.dumps(v)
            elif v is None:
                params[k] = ''
        return ' '.join([f"{k}={v}" for k, v in params.items() if v is not None])

    def dockerfiles(self, mod='mod'):
        """
        List all Dockerfiles in the specified path.
        
        Args:
            path (str): Path to search for Dockerfiles. Defaults to current directory.
            
        Returns:
            List[str]: List of Dockerfile paths.
        """
        dockerfiles = []
        path = c.dp(mod)
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.lower() == 'dockerfile':
                    dockerfiles.append(os.path.join(root, file))
        return dockerfiles

    def has_dockerfile(self, mod='mod'):
        """
        Check if a Dockerfile exists in the specified module path.
        
        Args:
            mod (str): Module name or path to check for Dockerfile.
            
        Returns:
            bool: True if a Dockerfile exists, False otherwise.
        """
        dockerfiles = self.dockerfiles(mod)
        return len(dockerfiles) > 0

    def dockerfile_path(self, mod='mod'):
        """
        Get the path to the Dockerfile in the specified module.
        
        Args:
            mod (str): Module name or path to check for Dockerfile.
            
        Returns:
            str: Path to the Dockerfile if it exists, otherwise None.
        """
        dockerfiles = self.dockerfiles(mod)
        # choose the shortest dockerfile path
        if len(dockerfiles) == 0:
            print(f'No Dockerfile found in {mod}')
            return None
        else: 
            print(f'Found {len(dockerfiles)} Dockerfiles in {mod}')
        dockerfiles = sorted(dockerfiles, key=len)
        return dockerfiles[0] if len(dockerfiles) > 0 else None


    def dockerfile(self, mod='mod'):
        return c.get_text(self.dockerfile_path(mod))

    def fork(self,  from_mod='mod', to_mod='mod'):
        return self.forward(f'make sure the source can have a docker container that has the contents of the current dockerfile --> {dockerfile}', src=src)


    def build(self,
              path: Optional[str] = None,
              tag: Optional[str] = None,
              mod = None,
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
        if mod:
            path = c.dirpath(mod)
        path = os.path.abspath(path or self.path)
        print(f'Building Docker image from {path} with tag {tag}')
        if os.path.isdir(path):
            assert os.path.exists(os.path.join(path, 'Dockerfile'))
        tag = tag or path.split('/')[-1]
        cmd = f'docker build -t {tag} .'
        if no_cache:
            cmd += ' --no-cache'
        return os.system('cd ' + path + ' && ' + cmd)


    def run(self,
            name : str = "commune",
            image: str = 'commune:latest',
            fn = 'fn',
            cwd: Optional = None, 
            cmd: str = None,
            volumes: Dict = None,
            gpus: Union[List, str, bool] = False,
            shm_size: str = '100g',
            sudo: bool = False,
            build: bool = False,
            net: Optional = None,  # 'host', 'bridge', etc.
            port: int = None,
            ports: Union[List, Dict[int, int]] = None,
            daemon: bool = True,
            remote: bool = False,
            entrypoint: Optional[str] = None,
            env: Optional[Dict] = None,
            working_dir = '/app',
            compose_path: str = None,
            restart: str = 'unless-stopped'
            ) -> Dict:
        """
        Generate and run a Docker container using docker-compose.
        """
        import yaml

        
        name = name or image.split('::')[0].replace('/', '_')
        
        # Build the service configuration
        service_config = {
            'image': image,
            'container_name': name,
            'restart': restart
        }
        
        # Handle command
        if cmd:
            service_config['entrypoint'] = f'bash -c "{cmd}"'
        
        # Handle network
        if net:
            service_config['network_mode'] = net
        
        # Handle GPU configuration
        if gpus:
            
            service_config['deploy'] = {
                'resources': {
                    'reservations': {
                        'devices': []
                    }
                }
            }

            if isinstance(gpus, list):
                for gpu in gpus:
                    service_config['deploy']['resources']['reservations']['devices'].append({
                        'driver': 'nvidia',
                        'device_ids': gpus,
                        'capabilities': ['gpu']
                    })
            elif isinstance(gpus, str):
                if gpus == 'all':
                    service_config['deploy']['resources']['reservations']['devices'].append({
                        'driver': 'nvidia',
                        'count': 'all',
                        'capabilities': ['gpu']
                    })
                else:
                    service_config['deploy']['resources']['reservations']['devices'].append({
                        'driver': 'nvidia',
                        'device_ids': [gpus],
                        'capabilities': ['gpu']
                    })
            elif gpus == True:
                service_config['deploy']['resources']['reservations']['devices'].append({
                    'driver': 'nvidia',
                    'count': 'all',
                    'capabilities': ['gpu']
                })
        
        # Configure shared memory
        if shm_size:
            service_config['shm_size'] = shm_size
        
        # Handle port mappings
        if self.server_exists(name):
            self.kill(name)
        if port:
            assert not c.port_used(port), f'Port {port} is already in use'
            ports = {port: port}
        
        if ports:
            if isinstance(ports, list):
                ports = {port: port for port in ports}
            service_config['ports'] = [f'{host}:{container}' for host, container in ports.items()]
        
        # Handle volume mappings
        if volumes:
            if isinstance(volumes, dict):
                service_config['volumes'] = [f'{c.abspath(k)}:{v}' for k, v in volumes.items()]
            elif isinstance(volumes, list):
                service_config['volumes'] = volumes
        
        # Handle environment variables
        if env:
            service_config['environment'] = env
         
        service_config['working_dir'] = working_dir

        if build: 
            if isinstance(build, dict):
                service_config['build'] = build
            elif isinstance(build, bool) and build:
                service_config['build'] = {'context': os.path.dirname(c.abspath(fn)), 'dockerfile': 'Dockerfile'}
            elif isinstance(build, str):
                service_config['build'] = {'context': build, 'dockerfile': 'Dockerfile'}
            service_config['image'] = f'{name}:latest'

        
        # Build the complete docker-compose configuration
        compose_config = {
            'services': {
                name: service_config
            }
        }

        
        # Add networks if needed
        if net and net != 'host':
            compose_config['networks'] = {
                net: {
                    'driver': 'bridge'
                }
            }
    
        # Write the docker-compose file
        cwd = cwd or os.getcwd()
        if compose_path is None:
            compose_path = cwd + '/docker-compose.yml' 
        compose_path = os.path.expanduser(compose_path)
        if not os.path.exists(compose_path):
            c.put_yaml(compose_path, compose_config)
        
        # Run docker-compose
        compose_cmd = ['sudo'] if sudo else []
        compose_cmd.extend(['docker-compose', '-f', compose_path, 'up'])
        # Run the container
        if bool(remote or daemon):
            compose_cmd.append('-d')
        command_str = ' '.join(compose_cmd)
        command_str = f'cd {cwd} && ' + command_str
        print(f'Running command: {command_str}')
        try:
            os.system(command_str)
        except Exception as e:
            print(f'Error running docker-compose: {e}')
        # remove the compose file if not daemon
        os.remove(compose_path)
        return  {'compose_file': compose_config['services'][name], 'cwd': cwd, 'name': name, 'status': 'running', 'command': command_str}

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
        return name in self.servers()
        
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
        if not self.exists(name):
            return {'status': 'not_found', 'name': name}
        name = self.name2process(name)
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
            for container in self.servers():
                self.kill(container, sudo=sudo, verbose=verbose)
            return {'status': 'all_containers_killed'}
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'servers': self.servers()}

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
             follow: bool = False, f = None,
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
        follow = f if f is not None else follow
        name = self.name2process(name)
        
        cmd = ['docker', 'logs']

        if tail:
            cmd.extend(['--tail', str(tail)])
        if since:
            cmd.extend(['--since', since])
        if follow:
            cmd.append('--follow')

        cmd.append(name)
        cmd = ' '.join(cmd)
        return os.system(cmd) if follow else c.cmd(cmd, verbose=verbose)

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
        return os.path.expanduser(f'~/.commune/pm/{path}')

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

    def container_stats(self, max_age=10, update=False, cache_dir="./docker_stats") -> pd.DataFrame:
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

        

    def get_port(self, name: str) -> Dict[int, int]:
        """
        Get the exposed ports of a container as a dictionary.
        
        Args:
            name (str): The container name (can use :: notation)
            
        Returns:
            Dict: Dictionary mapping host_port -> container_port
        """
        # Convert name format if needed
        container_name = self.name2process(name)
        
        # Get container inspection data
        try:
            inspect_output = c.cmd(f'docker inspect {container_name}', verbose=False)
            container_info = json.loads(inspect_output)[0]
            
            # Extract port bindings from HostConfig
            port_bindings = container_info.get('HostConfig', {}).get('PortBindings', {})
            
            # Convert port bindings to a simple dict format
            ports_dict = {}
            for container_port, host_configs in port_bindings.items():
                if host_configs:
                    # Extract port number from format like "8080/tcp"
                    container_port_num = int(container_port.split('/')[0])
                    # Get the host port from the first binding
                    host_port = int(host_configs[0]['HostPort'])
                    ports_dict = container_port_num
                    
            return ports_dict
            
        except Exception as e:
            c.print(f"Error getting ports for container {container_name}: {e}", color='red')
            return {}

    def namespace(self, search=None, max_age=None, update=True, **kwargs) -> dict:
        """
        Get a list of unique namespaces from container names.
        
        Returns:
            List[str]: List of unique namespaces
        """
        ip = '0.0.0.0'
        path = self.get_path('namespace.json')
        namespace = c.get(path, None, max_age=max_age, update=update)
        if namespace == None :
            containers = self.servers(search=search)
            namespace = {}
            for container in containers:
                port = self.get_port(container)
                namespace[container] =  ip + ':'+  str(port)
            c.put(path, namespace)
        return namespace

    def urls(self, search=None, mode='http') -> List[str]:
        return list(self.namespace(search=search).values())

    def start_docker_daemon(self):
        """
        Start the Docker daemon if it is not already running.
        
        Returns:
            str: Status message indicating whether the daemon was started or was already running.
        """

        try:
            c.cmd('systemctl start docker')
            return "Docker daemon started successfully."
        except Exception as e:
            return f"Error starting Docker daemon: {str(e)}"

    def is_docker_daemon_on(self):
        """
        Check if the Docker daemon is running.
        
        Returns:
            bool: True if the Docker daemon is running, False otherwise.
        """

        return not("Is the docker daemon running?" in c.cmd('docker info', verbose=False))


    def compose_files(self, mod = 'mod', depth=3) -> List[str]:
        """
        List all docker-compose files in the specified path.
        
        Args:
            path (str): Path to search for docker-compose files. Defaults to current directory.
            
        Returns:
            List[str]: List of docker-compose file paths.
        """
        compose_files = []
        path = c.dp(mod)
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.lower() in ['docker-compose.yml', 'docker-compose.yaml']:
                    compose_files.append(os.path.join(root, file))
        return compose_files
        

