


import streamlit as st
import os, sys
sys.path.append(os.getenv('PWD'))
import datasets
from copy import deepcopy
from commune import Module
class ClientModule(Module):
    registered_clients = {}

    def __init__(self, config=None ):
        Module.__init__(self, config=config,get_clients=False)
        self.register_clients(clients=self.include_clients)
    def get_default_clients(self):
        client_path_dict = dict(
        ipfs = 'commune.client.ipfs.module.IPFSModule',
        local = 'commune.client.local.module.LocalModule',
        s3 = 'commune.client.s3.module.S3Module',
        estuary = 'commune.client.estuary.module.EstuaryModule',
        pinata = 'commune.client.pinata.module.PinataModule',
        rest = 'commune.client.rest.module.RestModule',
        # ray='client.ray.module.RayModule'
        )
        return client_path_dict


    @property
    def client_path_dict(self):
        return self.get_default_clients()


    @property
    def default_clients(self):
        return list(self.get_default_clients().keys())

    def register_clients(self, clients=None):

        if isinstance(clients, list):
            assert all([isinstance(c,str)for c in clients]), f'{clients} should be all strings'
            for client in clients:
                self.register_client(client=client)
        elif isinstance(clients, dict):
            for client, client_kwargs in clients.items():
                self.register_client(client=client, **client_kwargs)
        else:
            raise NotImplementedError(f'{clients} is not supported')



    def get_client_class(self, client:str):
        assert client in self.client_path_dict, f"{client} is not in {self.default_clients}"
        return self.get_object(self.client_path_dict[client])

    def register_client(self, client, **kwargs):
        assert isinstance(client, str)
        assert client in self.default_clients,f"{client} is not in {self.default_clients}"

        client_module = self.get_client_class(client)(**kwargs)
        setattr(self, client, client_module )
        self.registered_clients[client] = client_module

    def remove_client(client:str):
        self.__dict__.pop(client, None)
        self.registered_clients.pop(client, None)
        return client
    
    delete_client = rm_client= remove_client
    
    def remove_clients(clients:list):
        output_list = []
        for client in clients:
            output_list.append(self.remove_client(client))
        return output_list

    delete_clients = rm_clients= remove_clients

    def get_registered_clients(self):
        return list(self.registered_clients.keys())

    @property
    def blocked_clients(self):
        return self.config.get('block', [])
    ignored_clients = blocked_clients

    @property
    def include_clients(self):
        include_clients = self.config.get('include', self.default_clients)
        block_clients = self.blocked_clients
        include_clients =  [c for c in include_clients if c not in block_clients]
        return include_clients

if __name__ == '__main__':
    import streamlit as st
    module = ClientModule()
    st.write(ClientModule._config())
    # st.write(module.__dict__)
