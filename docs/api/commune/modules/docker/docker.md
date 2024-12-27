# docker



Source: `commune/modules/docker/docker.py`

## Classes

### Docker



#### Methods

##### `build(self, path=None, tag=None, sudo=False, verbose=True, no_cache=False, env={})`



##### `chmod_scripts(self)`



##### `compose(self, path: str, compose: Union[str, dict, NoneType] = None, daemon: bool = True, verbose: bool = True, dash: bool = True, cmd: str = None, build: bool = False, project_name: str = None, cwd: str = None, down: bool = False)`



Type annotations:
```python
path: <class 'str'>
compose: typing.Union[str, dict, NoneType]
daemon: <class 'bool'>
verbose: <class 'bool'>
dash: <class 'bool'>
cmd: <class 'str'>
build: <class 'bool'>
project_name: <class 'str'>
cwd: <class 'str'>
down: <class 'bool'>
```

##### `compose_paths(self, path=None)`



##### `containers(self, sudo: bool = False)`



Type annotations:
```python
sudo: <class 'bool'>
```

##### `dashboard(self)`



##### `deploy(self, image: str, cmd: str = 'ls', volumes: List[str] = None, name: str = None, gpus: list = False, shm_size: str = '100g', sudo: bool = False, build: bool = True, ports: Dict[str, int] = None, net: str = 'host', daemon: bool = True, run: bool = True)`

Arguments:

Type annotations:
```python
image: <class 'str'>
cmd: <class 'str'>
volumes: typing.List[str]
name: <class 'str'>
gpus: <class 'list'>
shm_size: <class 'str'>
sudo: <class 'bool'>
build: <class 'bool'>
ports: typing.Dict[str, int]
net: <class 'str'>
daemon: <class 'bool'>
run: <class 'bool'>
```

##### `docker_compose(self, path='/home/bakobi/commune')`



##### `dockerfile(self, path='/home/bakobi/commune')`



##### `dockerfiles(self, path=None)`



##### `exists(self, name: str)`



Type annotations:
```python
name: <class 'str'>
```

##### `get_compose(self, path: str)`



Type annotations:
```python
path: <class 'str'>
```

##### `get_compose_path(self, path: str)`



Type annotations:
```python
path: <class 'str'>
```

##### `get_dockerfile(self, name)`



##### `image2id(self, image=None)`



##### `images(self, to_records=True)`



##### `install(self)`



##### `install_docker_compose(self, sudo=False)`



##### `install_gpus(self)`



##### `kill(self, name, sudo=False, verbose=True, prune=False)`



##### `kill_all(self, sudo=False, verbose=True)`



##### `kill_many(self, name, sudo=False, verbose=True)`



##### `log_map(self, search=None)`



##### `login(self, username: str, password: str)`



Type annotations:
```python
username: <class 'str'>
password: <class 'str'>
```

##### `logout(self, image: str)`



Type annotations:
```python
image: <class 'str'>
```

##### `logs(self, name, sudo=False, follow=False, verbose=False, tail: int = 2)`



Type annotations:
```python
tail: <class 'int'>
```

##### `name2compose(self, path=None)`



##### `name2dockerfile(self, path=None)`



##### `prune(self)`



##### `ps(self, search=None, df: bool = False)`



Type annotations:
```python
df: <class 'bool'>
```

##### `psdf(self, load=True, save=False, idx_key='container_id')`



##### `put_compose(self, path: str, compose: dict)`



Type annotations:
```python
path: <class 'str'>
compose: <class 'dict'>
```

##### `resolve_docker_compose_path(self, path=None)`



##### `resolve_docker_path(self, path=None)`



##### `resolve_dockerfile(self, name)`



##### `resolve_repo_path(self, path)`



##### `rm(self, name, sudo=False, verbose=True)`



##### `rm_container(self, name)`



##### `rm_image(self, image_id)`



##### `rm_images(self, search: List[str] = None)`



Type annotations:
```python
search: typing.List[str]
```

##### `rm_sudo(self, sudo: bool = True, verbose: bool = True)`

To remove the requirement for sudo when using Docker, you can configure Docker to run without superuser privileges. Here's how you can do it:
Create a Docker group (if it doesn't exist) and add your user to that group:
bash
Copy code
sudo groupadd docker
sudo usermod -aG docker $USER
return c.cmd(f'docker rm -f {name}', sudo=True)

Type annotations:
```python
sudo: <class 'bool'>
verbose: <class 'bool'>
```

##### `start_docker(self)`



##### `tag(self, image: str, tag: str)`



Type annotations:
```python
image: <class 'str'>
tag: <class 'str'>
```

