import ray
import os,sys
from commune.block.ray.queue import Queue
from commune.utils import dict_put,dict_get,dict_has,dict_delete
from copy import deepcopy
from commune import Module
"""

Background Actor for Message Brokers Between Quees

"""




class QueueServer(Module):

    default_actor_name = 'queue'
    default_config_path = 'ray.server.queue'
    def __init__(self,config=None, **kwargs):
        Module.__init__(self, config=config, **kwargs)
        self.queue = {}
        # self.topic2actorname = {}

    def topic2actorname(self, topic):
        root = self.actor_name
        if root == None:
            root = self.config.get('actor_name',
                                    self.config.get('root', 'queue'))

        return f'{root}.{topic}'

    def create_topic(self, topic:str,
                     maxsize:int=20,
                     refresh=True,
                      **kwargs):
        
        self.get_config()
        
        actor_name = kwargs.get('actor_name', self.topic2actorname(topic))

        if refresh :
            if self.actor_exists(topic):
                self.kill_actor(topic)
        queue = Queue(maxsize=maxsize, actor_options= dict( name=actor_name))
        self.queue[topic] = queue

        self.config['topic2actor'] = self.config.get('topic2actor', {})
        self.config['topic2actor'][topic] = actor_name


        self.put_config()
        return self.queue[topic] 


    @property
    def topic2actor(self):
        self.get_config()
        topic2actor =  self.config.get('topic2actor', {})
        new_topic2actor = {}
        for topic, actor in topic2actor.items():
            if self.actor_exists(actor):
                new_topic2actor[topic] = actor
        
        self.config['topic2actor'] = new_topic2actor

        self.put_config()
        return new_topic2actor


    def delete_topic(self,topic,
                     force=False,
                     grace_period_s=5,
                     verbose=False):

        queue = self.get_queue(topic)


        if isinstance(queue, Queue) :
            queue.shutdown(force=force, grace_period_s=grace_period_s)
            if verbose:
                print(f"{topic} shutdown (force:{force}, grace_period(s): {grace_period_s})")
        else:
            if verbose:
                print(f"{topic} does not exist" )
        # delete queue topic in dict
        self.queue.pop(topic)

    rm = delete = delete_topic

    def get_queue(self, topic, *args,**kwargs):
        # actor_name = self.topic2actorname(topic)
        return self.queue.get(topic)
    
    def topic_exists(self, topic, *args,**kwargs):
        return isinstance(self.queue.get(topic), Queue)

    exists = topic_exists




    def list_topics(self, **kwargs):
        return list(self.topic2actor.keys())
    ls = list_topics
    topics = property(list_topics)


    def put(self, topic, item, block=False, timeout=None, **kwargs):
        if not self.file_exists(topic):
            self.create_topic(topic=topic, **kwargs)
        
        try:
        
            self.get_queue(topic).put(item, block=block, timeout=timeout)
        except:
            pass
        del item


    def put_batch(self, topic, items, **kwargs):
        if not self.file_exists(topic):
            self.create_topic(topic=topic, **kwargs)
        return self.get_queue(topic).put_nowait_batch(items)


    def get_batch(self,topic, num_items=1):
        q = self.get_queue(topic)
        return q.get_nowait_batch(num_items = num_items)

    def get(self, topic, block=False, timeout=None, **kwargs):
        q = self.get_queue(topic)
        return q.get(block=block, timeout=timeout)

    def delete_all(self, force=True, *args, **kwargs):
        for topic in self.topics:
            self.delete_topic(topic, force=force, *args, **kwargs)

    rm_all = delete_all


    def size(self, topic):
        # The size of the queue
        return self.get_queue(topic).size()

    def empty(self, topic):
        # Whether the queue is empty.
        return self.get_queue(topic).empty()


    def full(self, topic):
        # Whether the queue is full.
        return self.get_queue(topic).full()

    def size_map(self):
        return {t: self.size(t) for t in self.topics}





class QueueClient(QueueServer):

    default_config_path = 'ray.server.queue'
    def __init__(self,config=None, **kwargs):
        Module.__init__(self, config=config, **kwargs)
        self.queue = {}
        
    def delete_topic(self,topic,
                     force=False,
                     grace_period_s=5,
                     verbose=False):

        

        queue_actor = self.get_queue(topic)
        return self.kill_actor(queue_actor)



    def delete_all(self):
        for topic in self.topics:
            self.delete_topic(topic, force=True)

    def size_map(self):
        return {t: self.size(t) for t in self.topics}
            
    def size(self, topic):
        # The size of the queue
        return self.get_queue(topic).size()

    def isempty(self, topic):
        # Whether the queue is empty.

        return self.get_queue(topic).empty()
    empty = isempty 

    def isfull(self, topic):
        # Whether the queue is full.
        return self.get_queue(topic).full()
    full = isfull

    def __del__(self):
        self.delete_all()

from functools import partial

class RayActorClient:
    def __init__(self, module):
        self.module =module
        for fn_key in module._ray_method_signatures.keys():

            def fn(self, fn_key,module, *args, **kwargs):
                ray_get = kwargs.pop('ray_get', False)
                object_id =(getattr(module, fn_key).remote(*args, **kwargs))
                if ray_get == True:
                    return ray.get(object_id)

                else:
                    return object_id

            setattr(self, fn_key, partial(fn, self, fn_key, module))
        
        

if __name__ == '__main__':
    import streamlit as st
    actor_name =  'queue_server'
    module = QueueServer.deploy(actor={'refresh':True, 'name': actor_name})
    
    x = ['fam']*1000
    st.write(ray.get(module.put_batch.remote('fam',[x]*1000)))
    st.write(ray.get(module.size.remote('fam')))
    # st.write(ray.get(module.get_batch.remote('fam', 2)))
    st.write(ray.get(module.getattr.remote('topic2actor')))
