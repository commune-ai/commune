import ray
from ray.util.queue import Queue
from commune import Module

"""

Background Actor for Message Brokers Between Quees

"""
import ray
from commune.block.ray.utils import kill_actor, create_actor

class RayRedisServer(Module):
    @staticmethod
    def set(key,message):
        return ray.global_worker.redis_client.set(key, message)
    @staticmethod
    def get(key):
        return ray.global_worker.redis_client.get(key)

    @classmethod
    def create_actor(cls,
                     actor_kwargs = {},
                     actor_name='redis_server',
                     detached=True,
                     resources={'num_cpus': 1, 'num_gpus': 0},
                     max_concurrency=3,
                     refresh=True,
                     return_actor_handle=False,
                     verbose = True):

        # this is a temp fix to get attributes from a given actor

        return create_actor(cls=cls,
                     actor_name=actor_name,
                     actor_kwargs=actor_kwargs,
                     detached=detached,
                     resources=resources,
                     max_concurrency=max_concurrency,
                     refresh=refresh,
                     return_actor_handle=return_actor_handle,
                     verbose=verbose)

    
