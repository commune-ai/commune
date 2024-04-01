# Docker Module

Docker is a class for interacting with Docker. This class provides methods for managing docker files, building Docker images, running Docker containers, and querying the state of existing containers.

## Methods

Here's a brief overview of the class methods:

- `dockerfile`: Returns the text of the Dockerfile at the given path.
- `resolve_repo_path`: Returns the absolute path to the repository.
- `docker_compose`: Returns the docker-compose file at the given path. 
- `build`: Builds a Docker container.
- `kill`: Kills a Docker container.
- `rm`: Removes a Docker container.
- `exists`: Checks if a Docker container exists.
- `rm_sudo`: Removes the requirement for sudo when using Docker.
- `containers`: Lists all Docker containers.
- `install`: Installs Docker based on a provided script.
- `images`: Returns a dataframe of all Docker images
- `rm_images`: Removes Docker images based on a given search criteria.
- `deploy`: Deploys a Docker container.
- `psdf`: Returns a dataframe with information about each running Docker container.
- `ps`: Lists running Docker containers.
- `get_dockerfile`: Resolves the docker file corresponding to a given name.
- `compose`: Executes Docker compose on a package.
- `logs`: Returns the logs of a Docker container.
- `tag`: Tags a Docker image.
- `login`: Login to a Docker registry.
- `logout`: Logout from a Docker registry.
- `dockerfiles`: Returns all Dockerfiles in a given path.
- `name2dockerfile`: Map container names to Dockerfiles.
- `dashboard`: (requires streamlit) Provides a web GUI for managing Docker containers.

## Usage

To use this module, import it into your Python script and instantiate a `Docker` object:

```python
from dockermodule import Docker
dock = Docker()
``` 

You can then use the methods of this class to interact with your Docker containers.
