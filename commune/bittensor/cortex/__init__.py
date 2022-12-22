
import commune

from . import utils
from .receptor import ReceptorPool as receptor_pool
from .trainer import trainer
from .proto import cortex_pb2 as proto
from .proto import cortex_pb2_grpc as grpc
from . import model

Module = commune.Module

get_annotations = Module.get_annotations
module = Module
launch =  Module.launch
import_module = Module.import_module
load_module = Module.load_module
import_object = Module.import_object
init_ray = ray_init=  Module.init_ray
start_ray = ray_start =  Module.ray_start
stop_ray = ray_stop=  Module.ray_stop
ray_initialized =  Module.ray_initialized
ray_context = Module.get_ray_context
list_actors = Module.list_actors
list_actor_names = Module.list_actor_names
get_parents = Module.get_parents
is_module = Module.is_module
run_command = Module.run_command