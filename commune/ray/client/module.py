
# Create Ocean instance
import streamlit as st
import os, sys
sys.path.append(os.getenv('PWD'))
from commune import Module
from functools import partial
import ray

class ClientModule(Module):
    override_attributes = False
    def __init__(self, actor=None, config=None, **kwargs):
        Module.__init__(self, config=config)
        
        actor = kwargs.get('server', actor)
        if actor == False:
            actor = self.config['server']
        if isinstance(actor, str):
            actor = self.get_actor(actor)
        elif isinstance(actor, dict):
            actor = self.get_module(**actor)
        elif isinstance(actor, ray.actor.ActorHandle):
            actor = actor

        actor_id = actor._ray_actor_id.hex()
        actor_name = None
        for a in Module.list_actors():
            if a['actor_id'] == actor_id:
                actor_name = a['name']

        self._actor_name = actor_name

        self.config['server'] = actor_name
        self.fn_signature_map = {}

        self.actor = actor
        
        self.parse()
        st.write(actor_name)


    @property
    def actor_id(self):
        return self.getattr('actor_id')

    @property
    def actor_name(self):
        return self._actor_name

    def getattr(self, ray_get=True, *args,**kwargs):
        object_id = self.actor.getattr.remote(*args,**kwargs)
        if ray_get:
            return ray.get(object_id)
        else:
            return object_id

    def setattr(self, ray_get=True, *args,**kwargs):
        object_id = self.actor.setattr.remote(*args,**kwargs)
        if ray_get:
            return ray.get(object_id)
        else:
            return object_id


    def submit(fn, *args, **kwargs):
        ray_get = kwargs.get('ray_get', True)
        ray_fn = getattr(self, fn)(*args, **kwargs)


    def submit_batch(fn, batch_kwargs=[], batch_args=[], *args, **kwargs):
        ray_get = kwargs.get('ray_get', True)
        ray_wait = kwargs.get('ray_wait', False)
        obj_id_batch = [getattr(self, fn)(*fn_args, **fn_kwargs) for fn_args, fn_kwargs in zip(batch_args, batch_kwargs)]
        if ray_get:
            return ray.get(obj_id_batch)
        elif ray_wait:
            return ray.wait(obj_id_batch)

    @property
    def ray_signatures(self):
        return self.actor._ray_method_signatures


    def parse(self):
        fn_ray_method_signatures = self.actor._ray_method_signatures
        for fn_key in fn_ray_method_signatures:
            def fn(self, fn_key,server, *args, **kwargs):
                
                ray_get = kwargs.pop('ray_get', True)
                is_batched = any([ k in kwargs for k in ['batch_kwargs', 'batch_args']]) 

                batch_kwargs = kwargs.pop('batch_kwargs',  [kwargs])
                batch_args = kwargs.pop('batch_args', [args])

                ray_fn = getattr(server, fn_key)

                object_ids =[ray_fn.remote(*args, **kwargs) for b_args,b_kwargs in zip(batch_args, batch_kwargs)]
                

   
                if ray_get == True:
                    output_objects =  ray.get(object_ids)

                else:
                    output_objects =  object_ids

                if is_batched:
                    return output_objects
                else:
                    assert len(output_objects) == 1
                    return output_objects[0]


            self.fn_signature_map[fn_key] = fn_ray_method_signatures
            setattr(self, fn_key, partial(fn, self, fn_key, self.actor))
            self.override_attributes = True 
    
    def __getattribute__(self, key):
        if key in ['actor', 'fn_signature_map']:
            return Module.__getattribute__(self, key)
        
        elif Module.__getattribute__(self, 'override_attributes'):


            if key in Module.__getattribute__(self, 'fn_signature_map'):
                return Module.__getattribute__(self, key)
            else:
                
                return Module.__getattribute__(self,'getattr')(key)


        return Module.__getattribute__(self, key)
    

    def __setattr__(self, *args, **kwargs):
        Module.__setattr__(self,*args, **kwargs)

    def __reduce__(self):
        deserializer = ClientModule
        serialized_data = (self.actor, self.config)
        return deserializer, serialized_data

    def __repr__(self):
        return 'ClientWrapped'
    def __str__(self):
        return 'ClientWrapped'
if __name__ == '__main__':
    module = ClientModule.deploy(actor=True)
    # st.write(module.get_functions(module))


    
