import fsspec
import os
from fsspec import register_implementation
import asyncio
import json
import pickle
import io
from datasets import load_dataset, Dataset
import os, sys
sys.path.append(os.getenv('PWD'))

import requests

from commune.client.local import LocalModule
from commune.client.ipfs import IPFSModule
from commune import Module
from commune.utils import try_n_times


# %% ../nbs/00_utils.ipynb 5
def parse_response(
    response, # Response object
):
    "Parse response object into JSON"
    
    if response.text.split('\n')[-1] == "":
        try:
            return [json.loads(each) for each in response.text.split('\n')[:-1]]
        
        except:
            pass

    try:
        return response.json()

    except:
        return response.text
    
# register_implementation(IPFSFileSystem.protocol, IPFSFileSystem)
# register_implementation(AsyncIPFSFileSystem.protocol, AsyncIPFSFileSystem)


class EstuaryModule(Module):

    def __init__(self, config=None):
        Module.__init__(self, config=config)
        self.api_key = self.get_api_key(api_key = self.config.get('api_key'))
        self.local =  LocalModule()
        self.ipfs = IPFSModule()
        self.url = self.config.get('url', 'https://shuttle-4.estuary.tech')

    def get_api_key(self, api_key=None):
        if api_key == None:
            api_key = self.config.get('api_key')
        api_key =os.getenv(api_key, api_key)
        if api_key != None:
            # if the api_key is a env variable
            return api_key
        else:
            # if the api key is just a key itself (raw)
            assert isinstance(api_key, str)
            return env_api_key


    def est_get_viewer(
        api_key: str=None # Your Estuary API key
    ):

        api_key = self.resolve_api_key(api_key)
        "View your Estuary account details"
        headers = {
        'Authorization': f'Bearer {api_key}',
        }

        response = requests.get(f'{self.url["get"]}/viewer', headers=headers)
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 6
    # list pins
    def list_pins(self,
        api_key: str=None # Your Estuary API key
    ):
        "List all your pins"

        api_key = self.resolve_api_key(api_key)
        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }

        response = requests.get(f'{self.url["get"]}/pinning/pins', headers=headers)
        return self.handle_response(response, return_fn=lambda x: x['results'] )


    ls_pins = list_pins

    # %% ../nbs/02_estuaryapi.ipynb 7
    # add pin


    def rm_all_pins(self):

        pins = self.list_pins()[1][0]['results']
        return [self.remove_pin(p['requestid']) for p in pins]

    def add_pin(self,
        file_name: str, # File name to pin
        cid: str, # CID to attach
        api_key: str=None # Your Estuary API key

    ):
        "Add a new pin object for the current access token."
        api_key = self.resolve_api_key(api_key)
        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }

        json_data = {
            'name': name,
            'cid': cid,
        }

        response = requests.post(f'{self.url["post"]}/pinning/pins', headers=headers, json=json_data)
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 8
    # get pin by ID
    def get_pin(self,
        pin_id: str, # Unique pin ID
        api_key: str=None # Your Estuary API key

    ):
        "Get a pinned object by ID"
        api_key = self.resolve_api_key(api_key)
        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }

        response = requests.get(f'{self.url["get"]}/pinning/pins/{pin_id}', headers=headers)
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 9
    # replace pin by ID
    def replace_pin(self,
        pin_id: str, # Unique pin ID
        api_key: str=None # Your Estuary API key

    ):
        api_key = self.resolve_api_key(api_key)
        "Replace a pinned object by ID"

        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }

        response = requests.post(f'{self.url["post"]}/pinning/pins/{pin_id}', headers=headers)
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 10
    # remove pin by ID
    def remove_pin(self,
        pin_id: str, # Unique pin ID
        api_key: str=None # Your Estuary API key

    ):
        "Remove a pinned object by ID"
        api_key = self.resolve_api_key(api_key)
        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }

        response = requests.delete(f'{self.url["get"]}/pinning/pins/{pin_id}', headers=headers)
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 12
    # create new collection
    def create_collection(self,
        name: str, # Collection name
        description: str='No Description', # Collection description
        handle_error:bool=True, 
        api_key: str= None # Your Estuary API key


    ):

        collection = self.get_collection(name)
        if collection !=None:
            return collection
        "Create new collection"
        api_key = self.resolve_api_key(api_key)

        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }

        json_data = {
            'name': name,
            'description': description,
        }

        response = requests.post(f'{self.url["get"]}/collections/create', headers=headers, json=json_data)
        return self.handle_response(response)

    add_collection = create_collection
    def rm_collection(self,
        collection: str, # Collection name
        api_key: str= None # Your Estuary API key

    ):

        "Create new collection"
        api_key = self.resolve_api_key(api_key)

        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }

        collection = self.get_collection(collection)
        if collection == None:
            return None
            
        assert isinstance(collection, dict)
        uuid = collection['uuid']

        response = requests.delete(f'{self.url["get"]}/collections/{uuid}', headers=headers)
        return self.handle_response(response)

    delete_collection = remove_collection = rm_collection


    # %% ../nbs/02_estuaryapi.ipynb 13
    # add content

    @property
    def name2collection(self):
        return {c['name']:c for c in self.list_collections()}

    @property
    def uuid2collection(self):
        return {c['uuid']:c for c in self.list_collections()}


    def collection_exists(self, collection):
        return bool(self.get_collection(collection))

    def get_collection(self, collection:str, handle_error=True, create_if_null=False):
        # get collection by name
        collection_maps = [self.uuid2collection, self.name2collection]

        for collection_map in collection_maps:
            if collection in collection_map:
                return collection_map[collection]


        if create_if_null:
            return self.create_collection(collection)
        if handle_error:
            return None
        else:
            raise Exception(f'{collection} does not exist in {list(collection_maps[0].keys())} uuids or {list(collection_maps[1].keys())} name')

    def add_content(self,
        collection_id: str, # Collection ID
        data: list, # List of paths to data to be added
        cids: list, # List of respective CIDs
        api_key: str= None, # Your Estuary API key

    ):
        "Add data to Collection"

        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }

        json_data = {
            'contents': data,
            'cids': cids,
            'collection': collection_id,
        }

        response = requests.post(f'{self.url["post"]}/collections/add-content', headers=headers, json=json_data)
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 14
    # list collections

    def list_collections(self,
        api_key: str=None # Your Estuary API key
    ):
        "List your collections"

        api_key = self.resolve_api_key(api_key)

        headers = {
        'Authorization': f'Bearer {api_key}',
        }

        response = requests.get(f'{self.url["get"]}/collections/list', headers=headers)
        
        
        
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 15
    # list collection content
    def list_collection_content(self,
        collection_id: str, # Collection ID
        api_key: str=None # Your Estuary API key


    ):
        api_key = self.resolve_api_key(api_key)

        "List contents of a collection from ID"

        headers = {
        'Authorization': f'Bearer {api_key}',
        }

        response = requests.get(f'{self.url["get"]}/collections/content/{collection_id}', headers=headers)
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 16
    # FS list content of a path
    def list_content_path(self,
        path: str='/dir1', # Path in collection to list files from
        collection: str='default', # Collection ID
        api_key: str=None # Your Estuary API key
    ):
        "List content of a path in collection"

        api_key = self.resolve_api_key(api_key)

        headers = {
        'Authorization': f'Bearer {api_key}',
        }

        collection_id = self.get_collection(collection)['uuid']

        params = {
            'col': collection_id,
            'dir': path
        }

        response = requests.get(f'{self.url["get"]}/collections/fs/list', params=params, headers=headers)
        return response

    # %% ../nbs/02_estuaryapi.ipynb 17
    # FS add content to path
    def add_content_path(self,
        collection_id: str, # Collection ID
        path: str, # Path in collection to add files to
        api_key: str=None # Your Estuary API key

    ):
        "Add content to a specific file system path in an IPFS collection"
        api_key = self.resolve_api_key(api_key)

        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }

        params = {
            'col': collection_id,
        }

        response = requests.post(f'{self.url["post"]}/collections/fs/add?col=UUID&content=LOCAL_ID&path={path}', params=params, headers=headers)
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 19
    # add client safe upload key
    def add_key(self,
        api_key:str, # Your Estuary API key
        expiry:str='24h' # Expiry of upload key
    ):
        "Add client safe upload key"

        headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json',
        }

        params = {
            'perms': 'upload',
            'expiry': expiry,
        }

        response = requests.post(f'{self.url["post"]}/user/api-keys', params=params, headers=headers)
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 20
    
    def add_files(self, paths:list, api_key:str=None):
        api_key = self.resolve_api_key(api_key)

        file2cid = {}
        for path in paths:
            assert isinstance(path, str)
            assert self.local.isfile(path_to_file)
            file2cid[file] = self.local.add_file(path)
        return file2cid

    def add_glob(self, path:str, api_key:str=None, return_type='dict'):
        api_key = self.resolve_api_key(api_key)
        file2cid = {}
        
        for path in self.local.glob(path):
            if self.local.isfile(path):
                #TODO: add asyncio or multi threading for call as it is IO bound
                file2cid[path] = self.add_file(path, return_cid= True)

        if return_type == 'dict':
            return file2cid
        elif return_type == 'list':
            return list(file2cid.values())
        else:
            raise NotImplementedError(f'only list and dict is supported but you did {return_type}')
    def add(self, path:str, api_key:str=None): 
        if self.local.isdir(path):
            return self.add_dir(path=path, api_key=api_key)
        elif self.local.isfile(path):
            return self.add_file(path=path, api_key=api_key)
            

    def add_dir(self, path:str,api_key:str=None, **kwargs):
        assert self.local.isdir(path), path
        return self.add_glob(path=path+'/**', **kwargs)  

    def add_file(self,
        path: str, # Path to file you want to upload
        api_key: str=None, # Your Estuary API key
        return_cid: bool = False
    ):
        "Upload file to Estuary"
        api_key = self.resolve_api_key(api_key)

        headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json',
        }

        files = {
            'data': open(path, 'rb'),
        }

        response = requests.post(f'{self.url["post"]}/content/add', headers=headers, files=files)
        response =  self.handle_response(response)

        if return_cid:
            return response['cid']

        
        if isinstance(response,dict):
            response['file'] = os.path.basename(path)
            response['size'] = self.ipfs.size(response['cid'])
            return response
        else:
            return response

    add_data = add_file
    # %% ../nbs/02_estuaryapi.ipynb 21
    # add CID
    def add_collection_cid(self,
        cid: str, # CID for file,
        path: str, # File name to add to CID
        collection:str='default',
        api_key: str=None, # Your Estuary API key

    ):
        "Use an existing IPFS CID to make storage deals."
        api_key = self.resolve_api_key(api_key)

        self.create_collection(collection)['uuid']

        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }
        
        coluid= self.get_collection(collection)['uuid']



        file_name = os.path.basename(path)

        json_data = {
            'name': file_name,
            'root': cid,
            'coluid': coluid,
            'collectionPath': path
        }

        response = requests.post(f'{self.url["get"]}/content/add-ipfs', headers=headers, json=json_data)
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 22
    # add CAR
    def add_car(self,
        path: str, # Path to file to store
        api_key: str=None, # Your Estuary API key
    ):
        "Write a Content-Addressable Archive (CAR) file, and make storage deals for its contents."

        headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json',
        }
        api_key = self.resolve_api_key(api_key)


        with open(path, 'rb') as f:
            data = f.read()

        response = requests.post(f'{self.url["post"]}/content/add-car', headers=headers, data=data)
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 23
    # make deal with specific provider
    def make_deal(self,
        content_id: str, # Content ID on Estuary
        provider_id: str, # Provider ID
        api_key: str=None # Your Estuary API key

    ):
        api_key = self.resolve_api_key(api_key)

        "Make a deal with a storage provider and a file you have already uploaded to Estuary"

        headers = {
        'Authorization': f'Bearer {api_key}',
        }

        json_data = {
            'content': content_id,
        }

        response = requests.post(f'{self.url["post"]}/deals/make/{provider_id}', headers=headers, json=json_data)
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 24
    # data by CID
    def view_data_cid(self,
        cid: str, # CID
        api_key: str=None, # Your Estuary API key

    ):
        "View CID information"
        api_key = self.resolve_api_key(api_key)

        headers = {
        'Authorization': f'Bearer {api_key}',
        }

        response = requests.get(f'{self.url["get"]}/content/by-cid/{cid}', headers=headers)
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 25
    # list data
    def list_data(self,
        api_key: str=None # Your Estuary API key
    ):
        api_key = self.resolve_api_key(api_key)

        headers = {
        'Authorization': f'Bearer {api_key}',
        }

        response = requests.get(f'{self.url["get"]}/content/stats', headers=headers)
        return self.handle_response(response)

    # list deals
    def list_deals( self,
        api_key: str=None # Your Estuary API key
    ):
        # list deals
        api_key = self.resolve_api_key(api_key)

        headers = {
        'Authorization': f'Bearer {api_key}',
        }

        response = requests.get(f'{self.url["get"]}/content/deals', headers=headers)
        return self.handle_response(response)

    # get deal status by id

    def resolve_api_key(self, api_key):
        if api_key == None:
            api_key = self.api_key
        assert isinstance(api_key, str)
        return api_key


    def get_deal_status(self,
        deal_id: str, # Deal ID,
    ):
        "Get deal status by id"

        headers = {
        'Authorization': f'Bearer {api_key}',
        }

        response = requests.get(f'{self.url["get"]}/content/status/{deal_id}', headers=headers)
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 28
    # get Estuary node stats
    def get_node_stats(self):
        "Get Estuary node stats"

        response = requests.get(f'{self.url["get"]}/public/stats')
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 29
    # get on chain deal data
    def get_deal_data(self):
        "Get on-chain deal data"

        response = requests.get(f'{self.url["get"]}/public/metrics/deals-on-chain')
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 30
    # get miner query ask

    def get_miner_ask(self,
        miner_id: str # Miner ID
    ):
        "Get the query ask and verified ask for any miner"

        response = requests.get(f'{self.url["get"]}/public/miners/storage/query/{miner_id}')
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 31
    # get failure logs by provider
    def get_failure_logs(self,
        miner_id: str # Miner ID
    ):
        "Get all of the failure logs for a specific miner"

        response = requests.get(f'{self.url["get"]}/public/miners/failures/{miner_id}')
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 32
    # get deal logs by provider
    def get_deal_logs(self,
        provider_id: str # Provider ID
    ):
        "Get deal logs by provider"

        response = requests.get(f'{self.url["get"]}/public/miners/deals/{provider_id}')
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 33
    # get provider stats
    def get_provider_stats(self,
        provider_id: str # Provider ID
    ):
        "Get provider stats"

        response = requests.get(f'{self.url["get"]}/public/miners/stats/{provider_id}')
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 34
    # list providers
    def list_providers(self):
        "List Estuary providers"

        response = requests.get(f'{self.url["get"]}/public/miners')
        return self.handle_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 36

    def get_data(self,
        cid: str, # Data CID
        path_name: str # Path and filename to store the file at
    ):
        "Download data from Estuary CID"

        url = f'{self.url["dweb"]}/{cid}'
        response = requests.get(url, allow_redirects=True)  # to get content
        with open(path_name, 'wb') as f:
            f.write(response.content)
        return self.handle_response(response)




    @property
    def tmp_root_path(self):
        return f'/tmp/commune/{self.id}'

    def get_temp_path(self, path='get_temp_path'):
        tmp_path = os.path.join(self.tmp_root_path, path)
        if not os.path.exists(self.tmp_root_path):
            self.local.makedirs(os.path.dirname(path), exist_ok=True)

        return tmp_path


    def save_model(self, model, path:str=None):


        # self.mkdir(path, create_parents=True)

        tmp_path = self.get_temp_path(path=path)
        model.save_pretrained(tmp_path)
        self.mkdirs(path)

        cid = self.force_put(lpath=tmp_path, rpath=path, max_trials=10)
        self.local.rm(tmp_path,  recursive=True)

        return cid

    def save_tokenizer(self, tokenizer, path:str='tmp'):


        # self.mkdir(path, create_parents=True)

        tmp_path = self.get_temp_path(path=path)
        tokenizer.save_pretrained(tmp_path)
        self.mkdirs(path)

        cid = self.force_put(lpath=tmp_path, rpath=path, max_trials=10)
        self.local.rm(tmp_path,  recursive=True)

        return cid

    def load_tokenizer(self, path:str='tmp'):
        tmp_path = self.get_temp_path(path=path)
        self.get(lpath=tmp_path, rpath=path )
        model = AutoTokenizer.from_pretrained(tmp_path)
        self.local.rm(tmp_path,  recursive=True)
        return model

    def load_model(self,  path:str='tmp'):
        tmp_path = self.get_temp_path(path=path)
        self.get(lpath=tmp_path, rpath=path )
        model = AutoModel.from_pretrained(tmp_path)
        # self.fs.local.rm(tmp_path,  recursive=True)
        return model

    def load_dataset(self, path='tmp', mode='huggingface'):
        tmp_path = self.get_temp_path(path=path)
        self.get(lpath=tmp_path, rpath=path )
        if mode in ['huggingface', 'hf', '🤗']:
            dataset = Dataset.load_from_disk(tmp_path)
            return dataset
            # self.fs.local.rm(tmp_path,  recursive=True)
        elif mode in ['activeloop', 'al']:
            raise NotImplementedError



    supported_dataset_modes = [
        'huggingface'
        'activeloop'
    ]

    def save_dataset(self, dataset=None, mode='🤗', return_type='dict', **kwargs):
        
        tmp_path = self.get_temp_path()
        self.local.makedirs(tmp_path, True)

        if mode in ['huggingface', 'hf', '🤗']:
            if dataset == None:
                load_dataset_kwargs = {}
                
                for k in ['path', 'name', 'split']:
                    v= kwargs.get(k)
                    assert isinstance(v, str), f'{k} is {v} but should be a string'
                    load_dataset_kwargs[k] = v
                dataset = load_dataset(**load_dataset_kwargs)
            dataset = dataset.save_to_disk(tmp_path)

        elif mode == 'activeloop':
            if dataset == None:
                path =  kwargs.get('path')
            else:
                raise NotImplementedError

        return self.add(path=tmp_path)



    def put_json(self, data, path='json_placeholder.pkl'):
        tmp_path = self.get_temp_path(path=path)
        self.local.put_json(path=tmp_path, data=data)
        cid = self.force_put(lpath=tmp_path, rpath=path, max_trials=10)
        self.local.rm(tmp_path)
        return cid

    def put_pickle(self, data, path='/pickle_placeholder.pkl'):
        tmp_path = self.get_temp_path(path=path)
        self.local.put_pickle(path=tmp_path, data=data)
        cid = self.force_put(lpath=tmp_path, rpath=path, max_trials=10)
        self.local.rm(tmp_path)
        return cid
    
    def get_pickle(self, path):
        return pickle.loads(self.cat(path))

    def get_json(self, path):
        return json.loads(self.cat(path))



    def save(self, lpath, rpath, max_trials=10):
        trial_count = 0
        cid = None
        while trial_count<max_trials:
            try:

                if self.local.isdir(lpath):
                    files = self.recursive_file_list(lpath)
                else:
                    files = [lpath]
                # for f in self.local.ls(lpath)
                cid= [self.add_data(f) for f in files]
                break
            except fsspec.exceptions.FSTimeoutError:
                trial_count += 1
                print(f'Failed {trial_count}/{max_trials}')

        return cid

    @property
    def id(self):
        return type(self).__name__ +':'+ str(hash(self))

    @property
    def name(self):
        return self.id

    @staticmethod
    def handle_response(response, return_fn=None):

        if response.status_code == 200:
            parsed_response = parse_response(response)
            output = None
            if len(parsed_response)==1 and \
                 isinstance(parsed_response, list):
                 parsed_response = parsed_response[0]
            else:
                parsed_response =  parsed_response

            if callable(return_fn):
                parsed_response = return_fn(parsed_response)
            
            return parsed_response
             
        else:
            return response

    @property
    def collection_names(self):
        return [c['name'] for c in self.list_collections()]


    def info(self, cid):
        
        for pin in self.list_pins():
            if pin['pin']['cid'] == cid:
                response = { k:pin['pin'][k] for k in ['cid', 'name']}
                response.update({k:v for k,v in self.ipfs.info(cid).items() if k in ['size', 'type']})
                return response
    @property
    def collection_uuids(self):
        return [c['uuid'] for c in self.list_collections()]


    def describe_state(self, mode='type'):
        if mode == 'type':
            return {k: type(v)for k,v in self.__dict__.items()}
        else:
            raise NotImplementedError(f'mode not supported {mode}')
        st.write(dataset.__dict__.keys())
        return dataset 



if __name__ == '__main__':
    import ipfspy
    import streamlit as st
    from commune.utils import *


    module = EstuaryModule()
    module.describe(streamlit=True, sidebar=True)
    from commune.utils import Timer




    def shard_dataset(dataset, shards=10, return_type='list'):
        '''
        return type options are list or dict
        '''
        if return_type == 'list':
            return [dataset.shard(shards, s) for s in range(shards)]
        elif return_type == 'dict':
             return DatasetDict({f'shard_{s}':dataset.shard(shards, s) for s in range(shards)})

        else:
            raise NotImplemented

    def recursive_file_list(self, path):

        output_files = [f for f in self.local.glob(path+'/**') if self.local.isfile(f)]

        return output_files

        