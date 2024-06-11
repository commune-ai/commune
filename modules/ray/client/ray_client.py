
# Create Ocean instance
import streamlit as st
import os, sys
from commune import Module
from functools import partial
import ray

class ClientModule(Module):
    
    protected_keys = ['actor', 'fn_signature_map']
    
    def __init__(self, actor: 'ray.actor'=None, **kwargs):
        self.attributes_parsed = False
        self.set_actor(actor)
        self.parse()

    def set_actor(self, actor):
        assert actor != None
        if isinstance(actor, str):
            actor = self.get_actor(actor)
        elif isinstance(actor, dict):
            actor = self.get_module(**actor)
        elif isinstance(actor, ray.actor.ActorHandle):
            actor = actor
        else:
            raise NotImplemented(actor)
        self.actor = actor

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


    def submit(self, fn, *args, **kwargs):
        ray_get = kwargs.get('ray_get', True)
        ray_fn = getattr(self, fn)(*args, **kwargs)


    def submit_batch(self, fn, batch_kwargs=[], batch_args=[], *args, **kwargs):
        ray_get = kwargs.get('ray_get', True)
        ray_wait = kwargs.get('ray_wait', False)
        obj_id_batch = [getattr(self, fn)(*fn_args, **fn_kwargs) for fn_args, fn_kwargs in zip(batch_args, batch_kwargs)]
        if ray_get:
            return ray.get(obj_id_batch)
        elif ray_wait:
            return ray.wait(obj_id_batch)

    def remote_fn(self, fn_key, *args, **kwargs):
        
        # is this batched fam
        ray_get = kwargs.get('ray_get', True)
        is_batched = any([ k in kwargs for k in ['batch_kwargs', 'batch_args']]) 

        batch_kwargs = kwargs.pop('batch_kwargs',  [kwargs])
        batch_args = kwargs.pop('batch_args', [args])

        ray_fn = getattr(self.actor, fn_key)

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



    def parse(self):
        self.fn_signature_map = {}
        for fn_key, fn_ray_method_signatures in self.actor._ray_method_signatures.items():
            self.fn_signature_map[fn_key] = fn_ray_method_signatures
            remote_fn = partial(self.remote_fn, fn_key)
            setattr(self, fn_key, remote_fn)
        
        self.attributes_parsed = True 
    
    
    def __getattribute__(self, key):
        if key in ClientModule.protected_keys:
            return Module.__getattribute__(self, key)
        
        elif Module.__getattribute__(self, 'attributes_parsed'):


            if key in Module.__getattribute__(self, 'fn_signature_map'):
                return Module.__getattribute__(self, key)
            else:
                
                return Module.__getattribute__(self,'getattr')(key)


        return Module.__getattribute__(self, key)
    

    def __setattr__(self, *args, **kwargs):
        Module.__setattr__(self,*args, **kwargs)


    def __repr__(self):
        return self.server_name
    def __str__(self):
        return self.server_name
if __name__ == '__main__':
    module = ClientModule.deploy(actor=True)
    # st.write(module.get_functions(module))


    
