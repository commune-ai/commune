
import docker

def docker_client():
    """
    Get a Docker client object.

    :return: A Docker client object.
    """
    return docker.Client(base_url='unix://var/run/docker.sock')
def docker_container_info(container_id):
    """
    Get information about a Docker container.

    :param container_id: The ID of the container.
    :return: A dictionary with information about the container.
    """
    try:
        return docker_client().inspect_container(container_id)
    except docker.errors.NotFound:
        return None