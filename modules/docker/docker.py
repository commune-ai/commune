
import os
import pandas as pd
from typing import List, Dict, Union, Optional, Any
import commune as c

class Docker(c.Module):
    def __init__(self):
        self.default_shm_size = '100g'
        self.default_network = 'host'
        
    def file(self, path: str = './') -> Union[str, Dict[str, str]]:
        """Get content of first Dockerfile found in path."""
        files = self.files(path)
        return c.get_text(files[0]) if files else {'msg': f'No Dockerfile found in {path}'}

    def files(self, path: str = './') -> List[str]:
        """Find all Dockerfiles in given path."""
        return [f for f in c.walk(path) if f.endswith('Dockerfile')]

    def build(self, 
             path: Optional[str] = None, 
             tag: Optional[str] = None, 
             sudo: bool = False, 
             verbose: bool = True, 
             no_cache: bool = False, 
             env: Dict[str, str] = {}) -> Dict[str, Any]:
        """Build Docker image from Dockerfile."""
        path = c.resolve_path(path)
        tag = tag or path.split('/')[-2]
        
        cmd = f'docker build -t {tag} .'
        if no_cache:
            cmd += ' --no-cache'
            
        result = c.cmd(cmd, sudo=sudo, env=env, cwd=os.path.dirname(path), verbose=verbose)
        return {'status': 'success', 'tag': tag, 'result': result}

    def run(self, 
            path: str = './',
            cmd: Optional[str] = None,
            volumes: Optional[Union[List[str], Dict[str, str], str]] = None,
            name: Optional[str] = None,
            gpus: Union[List[int], str, bool] = False,
            shm_size: str = '100g',
            sudo: bool = False,
            build: bool = True,
            ports: Optional[Dict[str, int]] = None,
            net: str = 'host',
            daemon: bool = True,
            cwd: Optional[str] = None,
            env_vars: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Run a Docker container with advanced configuration options.
        
        Args:
            path: Path to Dockerfile or image name
            cmd: Command to run in container
            volumes: Volume mappings
            name: Container name
            gpus: GPU configuration
            shm_size: Shared memory size
            sudo: Use sudo
            build: Build image before running
            ports: Port mappings
            net: Network mode
            daemon: Run in daemon mode
            cwd: Working directory
            env_vars: Environment variables
        """
        dcmd = ['docker', 'run']
        
        if daemon:
            dcmd.append('-d')
            
        dcmd.extend(['--net', net])
        
        # Handle GPU configuration
        if isinstance(gpus, list):
            dcmd.append(f'--gpus "device={",".join(map(str, gpus))}"')
        elif isinstance(gpus, str):
            dcmd.append(f'--gpus "{gpus}"')
            
        # Configure shared memory
        if shm_size:
            dcmd.extend(['--shm-size', shm_size])
            
        # Handle port mappings
        if ports:
            for host_port, container_port in ports.items():
                dcmd.extend(['-p', f'{host_port}:{container_port}'])
                
        # Handle volume mappings
        if volumes:
            if isinstance(volumes, str):
                volumes = [volumes]
            elif isinstance(volumes, dict):
                volumes = [f'{k}:{v}' for k, v in volumes.items()]
            for volume in volumes:
                dcmd.extend(['-v', volume])
                
        # Handle environment variables
        if env_vars:
            for key, value in env_vars.items():
                dcmd.extend(['-e', f'{key}={value}'])
                
        # Set container name
        if name:
            dcmd.extend(['--name', name])
            
        # Add image name
        dcmd.append(path)
        
        # Add command if specified
        if cmd:
            dcmd.extend(['bash', '-c', cmd])
            
        return {
            'cmd': ' '.join(dcmd),
            'cwd': cwd or os.getcwd()
        }

    def kill(self, name: str, sudo: bool = False, verbose: bool = True, prune: bool = False) -> Dict[str, str]:
        """Kill and remove a container."""
        c.cmd(f'docker kill {name}', sudo=sudo, verbose=verbose)
        c.cmd(f'docker rm {name}', sudo=sudo, verbose=verbose)
        if prune:
            self.prune()
        return {'status': 'killed', 'name': name}

    def kill_all(self, sudo: bool = False, verbose: bool = True) -> Dict[str, str]:
        """Kill all running containers."""
        for container in self.ps():
            self.kill(container, sudo=sudo, verbose=verbose)
        return {'status': 'all_containers_killed'}

    def images(self, to_records: bool = True) -> Union[pd.DataFrame, Any]:
        """List all Docker images."""
        text = c.cmd('docker images', verbose=False)
        rows = []
        cols = []
        
        for i, line in enumerate(text.split('\n')):
            if not line.strip():
                continue
            if i == 0:
                cols = [col.strip().lower().replace(' ', '_') for col in line.split() if col]
            else:
                rows.append([col.strip() for col in line.split() if col])
                
        df = pd.DataFrame(rows, columns=cols)
        return df.to_records() if to_records else df

    def logs(self, 
            name: str, 
            sudo: bool = False, 
            follow: bool = False, 
            verbose: bool = False, 
            tail: int = 100,
            since: Optional[str] = None) -> str:
        """Get container logs with advanced options."""
        cmd = ['docker', 'logs']
        
        if follow:
            cmd.append('-f')
        if tail:
            cmd.extend(['--tail', str(tail)])
        if since:
            cmd.extend(['--since', since])
            
        cmd.append(name)
        return c.cmd(' '.join(cmd), verbose=verbose)

    def prune(self, all: bool = False) -> str:
        """Prune Docker resources."""
        cmd = 'docker system prune -f' if all else 'docker container prune -f'
        return c.cmd(cmd)

    def stats(self, container: Optional[str] = None) -> pd.DataFrame:
        """Get container resource usage statistics."""
        cmd = f'docker stats --no-stream {container if container else ""}'
        output = c.cmd(cmd, verbose=False)
        return pd.read_csv(pd.StringIO(output), sep=r'\s+')

