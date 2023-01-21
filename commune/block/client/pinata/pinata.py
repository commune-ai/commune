import fsspec
import os
from fsspec import register_implementation
import asyncio
import json
import pickle
import io
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset
import os, sys
sys.path.append(os.getenv('PWD'))

import requests

from commune.client.local import LocalModule
from commune import Module

import streamlit as st


# register_implementation(IPFSFileSystem.protocol, IPFSFileSystem)
# register_implementation(AsyncIPFSFileSystem.protocol, AsyncIPFSFileSystem)

# with fsspec.open("ipfs://QmZ4tDuvesekSs4qM5ZBKpXiZGun7S2CYtEZRB3DYXkjGx", "r") as f:
#     print(f.read())


class PinataModule(Module):
    url = 'https://api.pinata.cloud'

    def __init__(self, config=None):
        Module.__init__(self, config=config)
        self.api_key = self.get_api_key(api_key = self.config.get('api_key'))
        self.url = self.config.get('url', f'{self.url}')
        self.local =  LocalModule()

    def get_api_key(self, api_key=None):
        if api_key == None:

            api_key = self.config.get('api_key')

        if api_key != None:
            # if the api_key is a env variable
            return api_key
        else:
            # if the api key is just a key itself (raw)
            assert isinstance(api_key, str)
            return api_key

    # %% ../nbs/03_pinataapi.ipynb 6
    def generate_apikey(self,
                        key_name:str, #Key name
                        pinlist:bool=False,#list pins
                        userPinnedDataTotal:bool=False, #total data stored
                        hashMetadata:bool=True, #metadata
                        hashPinPolicy:bool=False, #policy
                        pinByHash:bool=True, #pin cid
                        pinFileToIPFS:bool=True,#upload file to IPFS
                        pinJSONToIPFS:bool=True,#upload json to IPFS
                        pinJobs:bool=True,#see pin jobs
                        unpin:bool=True,#unpin ipfs cid
                        userPinPolicy:bool=True #establish pin policy

    ):

        url = f"{self.url}/users/generateApiKey"

        payload = json.dumps({
          "keyName": key_name,
          "permissions": {
            "endpoints": {
              "data": {
                "pinList": pinlist,
                "userPinnedDataTotal": userPinnedDataTotal
              },
              "pinning": {
                "hashMetadata": hashMetadata,
                "hashPinPolicy": hashPinPolicy,
                "pinByHash": pinByHash,
                "pinFileToIPFS": pinFileToIPFS,
                "pinJSONToIPFS": pinJSONToIPFS,
                "pinJobs": pinJobs,
                "unpin": unpin,
                "userPinPolicy": userPinPolicy
              }
            }
          }
        })
        headers = {
          'Authorization': f'Bearer {self.api_key}',
          'Content-Type': 'application/json'
        }

        response = requests.post(url, headers=headers, data=payload)

        return response


    # %% ../nbs/03_pinataapi.ipynb 9
    def list_apikeys(self):

        url = f"{self.url}/users/apiKeys"

        payload={}
        headers = {
          'Authorization': f'Bearer {self.api_key}'
        }

        response = requests.get(url, headers=headers, data=payload)

        return response


    # %% ../nbs/03_pinataapi.ipynb 12
    def revoke_apikey(self,revoke_apikey:str
    ):
        url = f"{self.url}/users/revokeApiKey"

        payload = json.dumps({
          "apiKey": revoke_apikey
        })
        headers = {
          'Authorization': f'Bearer {self.api_key}',
          'Content-Type': 'application/json'
        }

        response = requests.put(url, headers=headers, data=payload)

        return response


    # %% ../nbs/03_pinataapi.ipynb 15
    def upload_file(self,
                    name:str, #filename
                    fpaths:list, #filepaths
                    metadata:dict, #metadata
                    cid_version:str="1", #IPFS cid
                    directory:bool=False #upload directory
    ):

        pinataMetadata = dict({"name":name,"keyvalues":{}})
        pinataMetadata["name"] = name
        pinataMetadata["keyvalues"].update(metadata)

        pinataOptions = dict({"cidVersion":cid_version,"directory":directory})


        url = f"{self.url}/pinning/pinFileToIPFS"

        payload={"pinataOptions":json.dumps(pinataOptions),"pinataMetadata":json.dumps(pinataMetadata)}

        if directory:
            print("feature is not ready yet")

        files=[('file',(name,open(fpaths,'rb'),'application/octet-stream'))]

        headers = {
          'Authorization': f'Bearer {self.api_key}'
        }

        response = requests.post(url, headers=headers, data=payload, files=files)

        return response

    # %% ../nbs/03_pinataapi.ipynb 18
    def upload_jsonfile(self,
                    name:str, #filename
                    fpaths:list, #filepaths
                    metadata:dict, #metadata
                    cid_version:str, #IPFS cid
                    directory:bool=False #upload directory
    ):

        url = f"{self.url}/pinning/pinJSONToIPFS"

        payload = json.dumps({
          "pinataOptions": {
            "cidVersion": cid_version
          },
          "pinataMetadata": {
            "name": name,
            "keyvalues": metadata
          },
          "pinataContent": {"file":fpaths}
        })
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        response = requests.post(url, headers=headers, data=payload)

        return response

    # %% ../nbs/03_pinataapi.ipynb 21
    def pin(self,
            cid:str, #IPFS cid
            fn=None, #Name of file
            pinataMetadata=None #Add keys and values associated with IPFS CID
    ):

        url = f"{self.url}/pinning/pinByHash"

        payload = json.dumps({
          "hashToPin": cid,
          "pinataMetadata": {
            "name": fn,
            "keyvalues": pinataMetadata
          }
        })
        headers = {
          'Authorization': f'Bearer {self.api_key}',
          'Content-Type': 'application/json'
        }

        response = requests.post(url, headers=headers, data=payload)

        return response


    # %% ../nbs/03_pinataapi.ipynb 24
    def unpin(self,
              cid:str #IPFS CID
    ):

        url = f"{self.url}/pinning/unpin/{cid}"

        payload={}
        headers = {
          'Authorization': f'Bearer {self.api_key}'
        }

        response = requests.delete(url, headers=headers, data=payload)

        return response

    # %% ../nbs/03_pinataapi.ipynb 29
    def edit_metadata(self,
                  cid:str, #IPFS CID
                  name:str, #filename
                  metadata=None #Add keys and values associated with IPFS CID
    ):

        url = f"{self.url}/pinning/hashMetadata"

        pinataMetadata = dict({"name":name,"keyvalues":{}})
        pinataMetadata["keyvalues"].update(metadata)
        pinataMetadata["ipfsPinHash"] = cid

        payload = json.dumps(pinataMetadata)
        headers = {
          'Authorization': f'Bearer {self.api_key}',
          'Content-Type': 'application/json'
        }

        response = requests.put(url, headers=headers, data=payload)

        return response


    # %% ../nbs/03_pinataapi.ipynb 34
    def get_pinned_jobs(self,
                        params=None # filtering pinned jobs
    ):

        '''
        'sort' - Sort the results by the date added to the pinning queue (see value options below)
        'ASC' - Sort by ascending dates
        'DESC' - Sort by descending dates
        'status' - Filter by the status of the job in the pinning queue (see potential statuses below)
        'prechecking' - Pinata is running preliminary validations on your pin request.
        'searching' - Pinata is actively searching for your content on the IPFS network. This may take some time if your content is isolated.
        'retrieving' - Pinata has located your content and is now in the process of retrieving it.
        'expired' - Pinata wasn't able to find your content after a day of searching the IPFS network. Please make sure your content is hosted on the IPFS network before trying to pin again.
        'over_free_limit' - Pinning this object would put you over the free tier limit. Please add a credit card to continue pinning content.
        'over_max_size' - This object is too large of an item to pin. If you're seeing this, please contact us for a more custom solution.
        'invalid_object' - The object you're attempting to pin isn't readable by IPFS nodes. Please contact us if you receive this, as we'd like to better understand what you're attempting to pin.
        'bad_host_node' - You provided a host node that was either invalid or unreachable. Please make sure all provided host nodes are online and reachable.
        'ipfs_pin_hash' - Retrieve the record for a specific IPFS hash
        'limit' - Limit the amount of results returned per page of results (default is 5, and max is 1000)
        'offset' - Provide the record offset for records being returned. This is how you retrieve records on additional pages (default is 0)
        '''

        base_url = f'{self.url}/pinning/pinJobs/'

        header = {'Authorization': f'Bearer {self.api_key}'}

        response = requests.get(base_url, headers=header,params=params)

        return response

    # %% ../nbs/03_pinataapi.ipynb 37

    def get_pinned_files(self,params=None # Filter returned pinned files
    ):

        '''
        Query Parameters = params
        hashContains: (string) - Filter on alphanumeric characters inside of pin hashes. Hashes which do not include the characters passed in will not be returned.
        pinStart: (must be in ISO_8601 format) - Exclude pin records that were pinned before the passed in 'pinStart' datetime.
        pinEnd: (must be in ISO_8601 format) - Exclude pin records that were pinned after the passed in 'pinEnd' datetime.
        unpinStart: (must be in ISO_8601 format) - Exclude pin records that were unpinned before the passed in 'unpinStart' datetime.
        unpinEnd: (must be in ISO_8601 format) - Exclude pin records that were unpinned after the passed in 'unpinEnd' datetime.
        pinSizeMin: (integer) - The minimum byte size that pin record you're looking for can have
        pinSizeMax: (integer) - The maximum byte size that pin record you're looking for can have
        status: (string) -
            * Pass in 'all' for both pinned and unpinned records
            * Pass in 'pinned' for just pinned records (hashes that are currently pinned)
            * Pass in 'unpinned' for just unpinned records (previous hashes that are no longer being pinned on pinata)
        pageLimit: (integer) - This sets the amount of records that will be returned per API response. (Max 1000)
        pageOffset: (integer) - This tells the API how far to offset the record responses. For example,
        if there's 30 records that match your query, and you passed in a pageLimit of 10, providing a pageOffset of 10 would return records 11-20.
        '''

        base_url = f'{self.url}/data/pinList?'

        header = {'Authorization': f'Bearer {self.api_key}'}

        response = requests.get(base_url, headers=header,params=params)

        return response
    
    ls_pins = get_pinned_files
    ls = get_pinned_files


    # %% ../nbs/03_pinataapi.ipynb 40
    def get_datausage(self,params=None # Filter returned data usage statistics
    ):

        header = {'Authorization': f'Bearer {self.api_key}'}

        base_url = f'{self.url}/data/userPinnedDataTotal'

        response = requests.get(base_url, headers=header,params=params)

        return response

    @property
    def tmp_root_path(self):
        return f'/tmp/commune/{self.id}'

    def get_temp_path(self, path):
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

    def save_tokenizer(self, tokenizer, path:str=None):


        # self.mkdir(path, create_parents=True)

        tmp_path = self.get_temp_path(path=path)
        tokenizer.save_pretrained(tmp_path)
        self.mkdirs(path)

        cid = self.force_put(lpath=tmp_path, rpath=path, max_trials=10)
        self.local.rm(tmp_path,  recursive=True)

        return cid

    def load_tokenizer(self,  path:str):
        tmp_path = self.get_temp_path(path=path)
        self.get(lpath=tmp_path, rpath=path )
        model = AutoTokenizer.from_pretrained(tmp_path)
        self.local.rm(tmp_path,  recursive=True)
        return model

    def load_model(self,  path:str):
        tmp_path = self.get_temp_path(path=path)
        self.get(lpath=tmp_path, rpath=path )
        model = AutoModel.from_pretrained(tmp_path)
        # self.fs.local.rm(tmp_path,  recursive=True)
        return model
    @st.cache
    def load_dataset(self, path):
        tmp_path = self.get_temp_path(path=path)
        self.get(lpath=tmp_path, rpath=path )
        dataset = Dataset.load_from_disk(tmp_path)
        # self.fs.local.rm(tmp_path,  recursive=True)

        return dataset

    def save_dataset(self, dataset, path:str=None):
        tmp_path = self.get_temp_path(path=path)
        dataset = dataset.save_to_disk(tmp_path)
        cid = self.force_put(lpath=tmp_path, rpath=path, max_trials=10)
        # self.fs.local.rm(tmp_path,  recursive=True)
        return cid

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

    def force_put(self, lpath, rpath, max_trials=10):
        trial_count = 0
        cid = None
        while trial_count<max_trials:
            try:
                cid= self.put(lpath=lpath, rpath=rpath, recursive=True)
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

if __name__ == '__main__':
    import ipfspy
    import streamlit as st


    module = Pinata()
    st.write(module.name)



    # import torch

    st.write("Load Dataset")
    dataset = load_dataset('glue', 'mnli', split='train')
    st.write("Save Dataset")
    cid_pkl = module.save_dataset(dataset=dataset,path=module.tmp_root_path)
    st.write(cid_pkl)
    st.write("Pin to Pinata")
    cid = module.pin(cid_pkl,fn="Model")
    st.write(f"{cid},{cid.text}")
    # # cid = module.put_pickle(path='/bro/test.json', data={'yo':'fam'})
    # # st.write(module.get_pickle(cid))

    # st.write(module.ls('/dog'))
    # st.write(module.ls('/'))
    # st.write(module..get_object('/tmp/test.jsonjw4ij6u'))


    # st.write(module.local.ls('/tmp/bro'))
    # # st.write(module.add_data('/tmp/bro/state.json'))
    # st.write(module.get_node_stats())