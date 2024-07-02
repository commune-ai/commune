"""
 Implementation of Dataset using Asyncio and IPFS
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
import asyncio
import aiohttp

from copy import deepcopy
import json
from loguru import logger
import random
import os
import torch
from torch.utils.data.dataloader import DataLoader
from typing import Optional, Union, Dict, List, Any
import commune
from commune import Module
from commune.utils.dict import chunk

logger = logger.opt(colors=True)


class BittensorDataset(Module):
    """ Implementation for the dataset class, which handles dataloading from ipfs
    """

    ipfs_url = 'http://global.ipfs.opentensor.ai/api/v0'
    mountain_hash = 'QmSdDg6V9dgpdAFtActs75Qfc36qJtm9y8a7yrQ1rHm7ZX'
    
    
    def __init__(
            self, 
            batch_size: int = 32, 
            sequence_length: int = 256,
            min_block_size_bytes: int  = 10000,
            tokenizer: 'bittensor.tokenizer' = None,
            no_tokenizer: bool = False,
            max_hash_size:int = 10000000,
            num_workers: int = 1,
            datasets: Union[List[str], str] = None, 
            max_datasets: int = 10,
            max_directories: int = 100000,
            save_dataset : bool = True,
            load_dataset : bool = True,
            buffer_size:int = 1,
            buffer_calls_per_update: int = 1,
            download: bool = False,
            background: bool = True,
            min_hash_count : int = 850000,
            loop: Optional['asyncio.loop'] = None ,
            nest_asyncio: bool = True
            
            ):

        self.kwargs = locals()
        self.loop = loop if loop else self.get_event_loop()
        self.kwargs.pop('self')
        self.kwargs.pop('download')
        
        if nest_asyncio:
            commune.nest_asyncio()
        
        self.__dict__.update(self.kwargs)
        
        self.buffer_size = self.batch_size * self.buffer_size
        
        self.set_event_loop(loop=self.loop)
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]
        if self.datasets == 'default' or self.datasets == None:
            self.datasets = self.available_datasets

        self.datasets = self.datasets[:self.max_datasets]
        self.fetch_text_tasks = []
        self.sample_buffer = []


        self.save_dataset = save_dataset
        self.load_dataset = load_dataset
        self.set_tokenizer(tokenizer=self.tokenizer)

        # set the buffer
        self.set_buffer(buffer_size=buffer_size)

        # TODO: currently the number of batches is inert as this loop runs forever
        self.sample_count = 0
        self.batch_count = 0

        self.construct_text_corpus(datasets=self.datasets, load=self.load_dataset, save=self.save_dataset)

        
        if download:
            # Build the text corpus by fetching the hashes of the textfiles (Current Heirarchy)
        
            self.download_hashes(background=self.background, min_hash_count=self.min_hash_count)
      
      
    def set_tokenizer(self, tokenizer:'bittensor.tokenizer'=None)-> 'bittensor.tokenizer':
        try:
            import bittensor
        except RuntimeError as e:
            self.new_event_loop()
            import bittensor
        if tokenizer == None:
            tokenizer =   bittensor.tokenizer()
        self.tokenizer = tokenizer
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_idx = self.tokenizer(self.pad_token)['input_ids'][0]
        return self.tokenizer

    def construct_text_corpus(self, datasets:Optional[List[str]] = None, save:bool = False, load:bool = True) -> None :
        """ Building all of the datasets specified by getting each of their 
            text hashes from IPFS or local
        Args:
            datasets (List[str], optional):
                List of dataset names to include from the pile.
            save (bool, required):
                Save the dataset hashes locally.
            load (bool, required):
                Load the dataset hashes locally.
        """

        datasets = datasets if datasets else self.datasets
        all_text_file_metas = []
        dataset_hash_map = {}
        tasks = []
        self.dataset_byte_size_map = {d:0 for d in datasets}
        self.dataset_hash_size_map = {d:0 for d in datasets}
        
        # Gather dataset hashes async as their state is independent.
        for dataset in datasets:
            tasks += [self.async_build_single_dataset(dataset=dataset, save=save, load=load)]

        # Get the hashes asynchronously for each dataset.
        dataset_hashes = asyncio.run(asyncio.gather(*tasks))
        # Create a hash map of dataset -> text hashes.
        for k,v in zip(datasets, dataset_hashes):
            if len(v) > 0:
                dataset_hash_map[k] = v
                self.dataset_hash_size_map[k] = len(v)
        
        self.dataset_hash_map = dataset_hash_map

        # Flatten the hashes to a list of hashes.
        self.hash_dataset_map = {}
        self.all_text_file_metas = []
        for dataset_name,file_meta_list in dataset_hash_map.items():
            for fm in file_meta_list:
                if fm['Size'] >= self.min_block_size_bytes:
                    self.hash_dataset_map[fm['Hash']] = dataset_name
                    self.all_text_file_metas += [fm]
            
        self.dataset_size_map = {k:len(v) for k,v in self.dataset_hash_map.items()}
        # Ensure the hash list is not empty.
        assert len(self.all_text_file_metas) > 0

    async def async_build_single_dataset(self, dataset:str , save:Optional[bool]=False, load:Optional[bool]=True) -> List[dict] :
        """ Building a single dataset by fetching its text file metas ({Hash:str, Name:str, Size: int})
        Args:
            dataset (List[str], required):
                The name of the dataset.
            load (bool, optional):
                Load the dataset hashes locally
            save (bool, optional):
                Save the dataset hahses locally.

        Returns: 
            text_file_metas (List[Dict): 
                List of text file metas with the format of {Hash:String, Size:String, Name:String}.
        """
        # Hash to the meta file to avoid duplication in case we load two of the same file_meta.
        hash2file_meta = {}
        text_file_metas = []
        # If load is true, load the hashes, otherwise, fetch them from ipfs.
        if load:
            try:
                loaded_file_metas =  self.load_json(path=f'{dataset}/file_metas', default=[])
            except Exception as e:
                loaded_file_metas = []
                
            for file_meta in loaded_file_metas:
                hash2file_meta[file_meta['Hash']] = file_meta
            
            text_file_metas = list(hash2file_meta.values())
                        

        if len(text_file_metas) == 0:
        # Get the folder_hashes from the dataset.

            
            folder_hashes = (await self.get_folder_hashes(self.dataset2hash[dataset]))[:self.max_directories]
            # For each folder, get the text hashes.
            tasks = []
            for f in folder_hashes:
                tasks.append(self.get_folder_text_hashes(f, dataset=dataset))


            completed_task_results = await asyncio.gather(*tasks)
            # Some hashes are incomplete, ensure they have Size and Hash Field.
            for folder_text_file_metas in completed_task_results:
                for file_meta in folder_text_file_metas:
                    
                    if 'Size' in file_meta and 'Hash' in file_meta:
                        hash2file_meta[file_meta['Hash']] = file_meta   
                    
            text_file_metas = list(hash2file_meta.values())
        
        if save:
            self.save_json(path=f'{dataset}/file_metas', data=text_file_metas)
    
        # Calculate the size.
        self.dataset_byte_size_map[dataset]  = sum([fm['Size'] for fm in text_file_metas])

        self.print(f'Loaded {len(text_file_metas)} files from {dataset} with total size {self.dataset_byte_size_map[dataset]} bytes.')
        return text_file_metas

    def set_data_size(self, batch_size:Optional[int] = None, block_size:Optional[int] = None, sequence_length:Optional[int] = None,  min_block_size_bytes:Optional[int]= None, buffer_size:Optional[int]=None) -> None:
        r""" 
        Update the size of data (batch_size, sequence_length, min_block_size_bytes) that we need.

        Args: 
            batch_size (int, optional):
                The batch_size of data that should be produced by dataloader.

            sequence_length (int, optional):
                The number of tokens for each sample.

            min_block_size_bytes (int, optional):
                The min_block_size_bytes of data in bytes that should be produced by dataloader. 

            buffer_size(int, optional):
                The size of the buffer. 
        """

        def check_valid(size:int):
            r""" 
            Check if the size is a valid positive integer, if not, return False.
            """
            if (not isinstance(size, int)) or size <= 0:
                return False
            else:
                return True
        
        if check_valid(batch_size):
            self.batch_size = batch_size
            self.__infinite_dataset_iterator = None

        if check_valid(sequence_length):
            self.sequence_length = sequence_length

        if check_valid(block_size):
            logger.warning('The block size represents the seqeunce length and will be depracted')
            self.sequence_length = sequence_length
    
        if check_valid(min_block_size_bytes):
            self.min_block_size_bytes = min_block_size_bytes

        if check_valid(buffer_size):
            self.set_buffer(buffer_size= buffer_size)

    def set_buffer(self, buffer_size:int) -> None:
        """
        Set the buffer and ensure it is valid.

        Args:
            buffer_size (int, required):
                The size of the sample buffer.
        """
        if not hasattr(self, 'sample_buffer'):
            self.sample_buffer = []

        self.buffer_size = buffer_size 

        # If the buffer is smaller than the current buffer, trim it to match the new size.
        if len(self.sample_buffer) > self.buffer_size:
            self.sample_buffer = self.sample_buffer[:self.buffer_size]
            
            
    def suggest_samples(self, sample_size:int, loaded_fraction = 1.0):
        suggest_samples = []
        suggest_samples += random.sample(self.saved_hashes, int(sample_size * loaded_fraction))
        return suggest_samples
    
    async def async_generate_sample(self)-> List[str]:
        '''
        Checks the sample buffer, and builds it if it is empty

        Returns:
            self.sample_buffer (List[str]): 
                The sample buffer.
        '''
        # See if there is free space, if so, add jobs to fill the free space with samples.
        buffer_free_space = self.buffer_size - len(self.sample_buffer) 
                
        if buffer_free_space > 0  :
            
            # Sample the file_metas randomly.
            sample_cat_params_list = self.suggest_samples(buffer_free_space)
   

            # Build the asyncio jobs.
            self.fetch_text_tasks += [asyncio.create_task(self.fetch_text(file_meta=sample_cat_params, offset=0, length=self.max_hash_size, load=True, save=False)) for sample_cat_params in sample_cat_params_list]
            
            # This currently synchronytes on all of the self.fetch_text_tasks, completing when they all are finished.
            finished_tasks, running_tasks  = await asyncio.wait(self.fetch_text_tasks) 
            
            self.fetch_text_tasks = list(running_tasks)
            finished_tasks = list(finished_tasks)

            # Add the finished task results into the buffer.
            for finished_task in finished_tasks:
                sample = finished_task.result()
                if sample == None:
                    continue
                self.sample_buffer += [sample]

        # Randomly sample the text file from the buffer.
        random_idx = random.randint(0,len(self.sample_buffer)-1)

        raw_chunk = self.sample_buffer[random_idx]


        # Increment the counters.
        self.sample_count += 1
        self.batch_count += self.sample_count //  self.batch_size


        if self.min_block_size_bytes < len(raw_chunk):
            start_idx = random.randint(0, len(raw_chunk) - self.min_block_size_bytes)
        else:
            start_idx = 0
        
        end_idx = start_idx + self.min_block_size_bytes
        sample = raw_chunk[start_idx:end_idx]

        if (self.batch_count) >= self.buffer_calls_per_update:
            self.sample_count = 0 
            self.sample_buffer = self.sample_buffer[self.buffer_calls_per_update*self.batch_size:]
            self.batch_count = 0
        
        return sample

    def __getitem__(self, idx: Optional[int] = None, *args, **kwargs) -> Union[List[str], torch.tensor]:
        return asyncio.run(self.__async_getitem__(idx=idx, *args, **kwargs))

    def sample(self, batch_size:int=None, sequence_length:int = None, no_tokenizer:bool = None, task:str = None):
        batch_size = batch_size if batch_size else self.batch_size
        sequence_length = sequence_length if sequence_length else self.sequence_length
        no_tokenizer = no_tokenizer if no_tokenizer else self.no_tokenizer
        getitem_jobs = asyncio.gather(*[self.__async_getitem__(sequence_length=sequence_length, no_tokenizer=True) for i in range(batch_size)])
        sample_text = asyncio.run(getitem_jobs)
        output = {}
        if no_tokenizer:
            output['text'] =sample_text
        else:
            output['input_ids'] =self.tokenizer(sample_text, max_length=sequence_length, truncation=True, padding="max_length", return_tensors="pt")["input_ids"]
        
        # include the targets for causal language modeling
        if task == True:
            task = 'causallm'
        if task == None:
            pass
        elif task in ['causallm']:
            output['targets'] = self.tokenizer(sample_text, max_length=sequence_length, truncation=True, padding="max_length", return_tensors="pt")["input_ids"]
            
        return output

    __next__ = sample
    async def __async_getitem__(self, idx: Optional[int] = None, sequence_length:int=None, no_tokenizer:bool = None) -> Union[List[str], torch.tensor]:
        '''
        Sample from the sample_buffer via self.async_generate_sample. This fetches a random block of text
        with a size of self.min_block_size_bytes in bytes.
        Args:
            idx (int):
                Sample index of dataset.
            
        Returns:
            output (Union[str, torch.tensor])
        '''
        sequence_length = sequence_length if sequence_length else self.sequence_length
        no_tokenizer = no_tokenizer if no_tokenizer else self.no_tokenizer

        # only sample if the buffer is less than the buffer_size

        raw_text = await self.async_generate_sample()

        # Decode the bytes into a string.


        # If there is no tokenizer specified return text with the seqeunce length being the number of " " split elements.

        if no_tokenizer:
            raw_text =raw_text.split()
            output = raw_text[:sequence_length]
            remainder = sequence_length - len(output)

            if remainder > 0:
                # left side padding
                output = [self.pad_token]*remainder + output 

            output = ' '.join(output)
        else:
            output = self.tokenizer(raw_text, max_length=sequence_length, truncation=True, padding="max_length", return_tensors="pt")["input_ids"]
            output = output.to(torch.long).squeeze(0) #  [1,seq_len] -> [seq_len]

        return output
    
    
    async def get_dataset_hashes(self)-> List[dict]:
        '''
        Get the hashes representing the root of each dataset
        
        Returns
            response (dict):
            
        '''
        mountain_meta = {'Name': 'mountain', 'Folder': 'meta_data', 'Hash': self.mountain_hash}
        response = await self.api_post( 'object/get',  params={'arg': mountain_meta['Hash']}, return_json= True)
        response = response.get('Links', None)
        return response

    async def get_folder_hashes(self, file_meta:dict) -> List[str]:
        '''
        Get the folder hashes from the dataset.

        Args:
            file_meta (dict):
                File meta contianing the hash and name of the link.
        Returns folder_hashes (List[str])
        
        '''

        links = (await self.get_links(file_meta))
        
        # Build the tasks to fetch the links of the folder.
        unfinished = [asyncio.create_task(self.api_post('object/get', params={'arg':link['Hash']}, return_json=True)) for link in links]
        folder_hashes = []
        just_links = []

        # Gather results until all tasks are finished.
        while len(unfinished)>0:
            finished, unfinished = await asyncio.wait(unfinished, return_when=asyncio.FIRST_COMPLETED)
            for res in await asyncio.gather(*finished):
                folder_hashes.extend(res.get('Links'))
        
        # Sometimes, the folder_hashes are empty with some datasets.
        # This means the root links are the folder links.
        # TODO (for constructing text corpus): We need the root links to be 1 level for more consistancy.
        if len(folder_hashes) == 0:
            folder_hashes = links

        return folder_hashes
    
    @classmethod
    def download(cls, *args, **kwargs):
        kwargs['download'] = True
        self = cls(*args, **kwargs)
        
        
    @classmethod
    def run_task(cls,
                     fn = 'download',
                     jobs = 2,
                     *args, 
                     **kwargs):
        
        module_path = cls.module_path()
        prefix = f'task.{module_path}' 
        for job in range(jobs):
            name = f'{prefix}.{job}'
            
            cls.launch(fn='download', name=name, *args, **kwargs)
    
    def download_hashes(self,
                 chunk_size:int=100, 
                 background_thread:bool=False, 
                 ignore_error:bool =True, 
                 min_hash_count: int = 10000, 
                 background:bool = True, 
                 verbose_rate = 1):
        
        if background:
            thread_fn_kwargs = dict(locals())
            thread_fn_kwargs.pop('self', None)
            thread_fn_kwargs['background'] = False
            from threading import Thread
            self.download_thread = Thread(target=self.download, kwargs= thread_fn_kwargs, daemon=True, name='IPFS Download')
            self.download_thread.start()   
            
    
        loop = self.get_event_loop()
        
        file_meta_chunks = chunk(self.unsaved_hashes, chunk_size=chunk_size)
        
        random.shuffle(file_meta_chunks)
        
        total_hash_count = len(self.all_text_file_metas)
        fail_count = 0
        for i,  file_meta_chunk in enumerate(file_meta_chunks):
            if i % verbose_rate == 0:
                total_hash_count = len(self.all_text_file_metas)
                # if total_hash_count < min_hash_count:
                #     print(f'Not enough hashes to download. {total_hash_count} < {min_hash_count}')
                #     return
                num_saved_hashes = len(self.get_saved_hashes())
                if num_saved_hashes > min_hash_count: 
                    break
                
                self.print(f'{i} hashes downloaded -> Total Saved Hashes {num_saved_hashes}/{total_hash_count} fails: {fail_count}')
            # Build the asyncio jobs.
            try:
                loop.run_until_complete(asyncio.gather(*[self.fetch_text(file_meta=file_meta, offset=0, length=self.max_hash_size, load=False) for file_meta in file_meta_chunk ]))
            except Exception as e:
                fail_count += 1
                if ignore_error:
                    print(e)
                else:
                    raise(e)
        # This currently synchronytes on all of the self.fetch_text_tasks, completing when they all are finished.
        # finished_tasks, running_tasks  = await asyncio.wait(self.fetch_text_tasks) 
            
    last_time_saved_hashes = 0
    
    
    def get_saved_hashes(self, update:bool = True):
        
        _saved_hashes = []
        for dataset in self.datasets:
            hash_urls = self.glob(f'saved_file_metas/{dataset}/*')
            
            for hash_url in hash_urls:
                _saved_hashes += [{'Hash': hash_url.split('/')[-1]}]
        if update:
            self._saved_hashes = _saved_hashes
            
            
        return _saved_hashes
    
    @property
    def saved_hashes(self) -> List[Dict[str, dict]]:
        
        if not hasattr(self, '_saved_hashes'):
            self._saved_hashes = self.get_saved_hashes()
                    
            #  = {self.hash_dataset_map[h['Hash'].split('.')[0]]: h for h in self._saved_hashes}
            

        return self._saved_hashes


    def get_saved_hashes(self, update:bool = True):
        
        _saved_hashes = []
        for dataset in self.datasets:
            hash_urls = self.glob(f'saved_file_metas/{dataset}/*')
            
            for hash_url in hash_urls:
                _saved_hashes += [{'Hash': hash_url.split('/')[-1]}]
        if update:
            self._saved_hashes = _saved_hashes
        return _saved_hashes
    
    
    @property
    def num_hashes(self) -> int:
        return len(self.saved_hashes)
        
    @property
    def unsaved_hashes(self) -> List[str]:
        
        if not hasattr(self, '_unsaved_hashes'):
            hash_dataset_map = deepcopy(self.hash_dataset_map)
            for k in self.saved_hashes:
                
                hash_dataset_map.pop(k['Hash'], None) 
            unsaved_hashes = [{'Hash': h} for h in list(hash_dataset_map.keys())]
        
        return unsaved_hashes
        
            
    async def fetch_text(self, file_meta:dict, offset:int=0, length:int=None, save:bool = True, load:bool = True ):
        
        
        if isinstance(file_meta, str):
            file_meta = {'Hash': file_meta}
        
        length = length if length else self.max_hash_size
        cid = file_meta['Hash']
        cid = cid.split('.')[0]
        
        dataset = self.hash_dataset_map.get(cid, None)
        if dataset is None:
            return None
        
        path=f'saved_file_metas/{dataset}/{cid}'
        
        try:
            response = self.get_json(path=path, handle_error=True, default={}) if load else {}
        except json.decoder.JSONDecodeError:
            response = {}
            
        assert isinstance(response, dict)
        
        if 'text'in response:
            return response['text']
        
        else:
            #   print(f'Fetching {cid}')
            response  = await self.cat(cid=cid, offset=offset, length=length)
            try:
                # decode the response.
                response = response.decode()
            except UnicodeDecodeError as e:
                # fixes the issue with the ipfs cat endpoint returning a non utf-8 encoded string.
                response = str(response[2:-1])
            if save:
                file_meta['text'] = response
                self.put_json(path=path, data= file_meta)
        
            return response
    
    
    
    async def cat(self,
                  cid:str,
                  offset:int = 0,
                  length:int = 1000,
                  headers: dict=  None)->bytes:
        '''
        Cat endpoint.
        Args:
            cid (str):
                CID of the object.
            offset (int):
                The offset in bytes.
            length  (int):
                The length in bytes.
            
        Returns:
            response (bytes):
                The response from the cat call.
                
        '''
        
        headers = headers if headers else {}
        
        
        params = dict(arg=cid, offset=int(offset),lenght=int(length))
        headers = {}
        response = await self.api_post('cat', params=params, headers=headers, chunk_size=10000000, num_chunks=1)
        

        return response

    async def get_folder_text_hashes(
                                    self, 
                                    file_meta:dict, 
                                    dataset:str, 
                                    max_chunks:int = 1, 
                                    chunk_size:int = 1e10) -> List[Dict[str, Union[str, int]]]:
        """
        Get text hashes from a folder

        Args:
            file_meta (dict):
                File meta contianing the hash and name of the link.
            dataset (str):
                The name of the dataset for self.dataset_hash_map.
            max_chunks (int): 
                Max number of chunks to call when fetching file metas.
        
        Returns 
            text_file_metas (List[Dict[str, Union[str, int]]):
                List of the text file_metas of the folder.
        """
        text_file_metas = []
        
        for chunk_i in range(max_chunks):
            
            try:
                data = await self.cat(file_meta['Hash'], offset=chunk_i*chunk_size ,length=chunk_size)
            except Exception as e:
                self.print(e, color='red')
                continue
            
            
            
            if data == None:
                continue
            hashes = ['['+h + '}]'for h in data.decode().split('},')]
            for i in range(len(hashes)-1):
                try:
                    decoded_hash = json.loads(hashes[i+1][1:-1])
                    decoded_hash_size_bytes = decoded_hash.get('Size', 0)
                    if decoded_hash_size_bytes > 0:
                        self.dataset_byte_size_map[dataset] += decoded_hash_size_bytes
                        text_file_metas.append(decoded_hash)
                except json.JSONDecodeError:
                    pass
                except AttributeError as e:
                    continue
                
                hashes[i] ='{'+ hashes[i+1] + '}'

        return text_file_metas

    async def get_links(self, file_meta:dict) -> List[dict]:
        '''
        Get Links from file_meta

        Args
            file_meta (dict, required): 
                Dictionary containing hash and name of root link.
        '''
        response = await self.api_post( 'object/get',  params={'arg': file_meta['Hash']}, return_json= True)
        response_links = response.get('Links', [])
        return response_links

    async def api_post(
                    self, 
                    endpoint:str, 
                    params:Optional[Dict[str, Any]] = {}, 
                    headers:Optional[Dict[str, Any]] = {}, 
                    return_json:Optional[bool] = False,  
                    content_type:Optional[str] = None, 
                    chunk_size:Optional[int] = 1024, 
                    num_chunks:Optional[int] = None, 
                    sock_connect:Optional[int]=2, 
                    sock_read:Optional[int]=2) -> Union[Dict, 'aiohttp.Response', bytes]:
        '''
        Async api post to ipfs server.

        Args:
            endpoint (str):
                Endpoint path with such that path is "self.ipfs_url/{endpoint}".
            params (Dict[str, Any], optional):
                Params for api request.
            headers (Dict[str, Any], optional): 
                Headers for api request.
            return_json (bool, optional): 
                Return repsonse as json.
            content_type (str, optional):
                Content type of request.
            chunk_size (int, optional):
                Chunk size of streaming endpoint.
            num_chunks (int, optional):
                Number of chunks to stream.
            sock_connect (int, optional):
                The timeout for connecting to a socket.
            sock_read (int, optional):
                The timeout for reading a socket.
        Returns:
            return_result (Union[Dict, 'aiohttp.Response', bytes]):
                The result of the response. 
                    - Dictionary if return_json = True. 
                    - Bytes if num_chunks > 0
                    - aiohttp.Response if num_chunks == 0 and return_json == False
        '''
        url = os.path.join(self.ipfs_url, endpoint)
        return_result = None
        
        timeout = aiohttp.ClientTimeout(sock_connect=sock_connect, sock_read=sock_read)
        
        async with aiohttp.ClientSession( timeout=timeout) as session:
            async with session.post(url,params=params,headers=headers) as res:
                # Return a Json of the response.
                if return_json: 
                    return_result = await res.json(content_type=content_type)
                else:
                    return_result = res

                # If num_chunks is not None, iterate through the chunks of chunk_size.
                if num_chunks:
                    return_result = b''
                    async for data in res.content.iter_chunked(chunk_size):
                        return_result += data
                        num_chunks-= 1
                        if num_chunks == 0:
                            break
        return return_result

    @property
    def available_datasets(self) -> List[str]:
        '''
        List of available datasets.

        Retuns:
            List of available datasets.
        '''
        return list(self.dataset2hash.keys())

    @property
    def dataset_hashes(self) -> List[str]:
        '''
        Return the dataset hashes:

        Returns
            self._dataset_hashes (List[str]):
                A list of the dataset hashes
        '''
        # This avoids us from having to call this multiple times from IPFS.
        if not hasattr(self, '_dataset_hashes'):
            self._dataset_hashes = asyncio.run(self.get_dataset_hashes())
        return self._dataset_hashes

    @property
    def dataset2hash(self) -> Dict:
        '''
        Dictionary to hash
        '''
        
        return {v['Name'].replace('.txt', '') :v for v in self.dataset_hashes}
    
    @property
    def dataset_size(self) -> int:
        '''
        The size of the dataset in bytes.
        '''
        return sum(list(self.dataset_byte_size_map.values()))


    def dataloader(self, epoch_length:Optional[int] = 100) -> DataLoader:
        """ 
        Creates a torch dataloader out of a subclass of this class.

        Args:
            epoch_length (int, optional): 
                
                The epoch length of the miner. If this length is not set or if it is larger than the dataset,
                then a dataloader for the entire dataset is returned. Otherwise, a dataloader for a subset of the dataset of epoch_length
                is returned. 

        Returns:
            torch.utils.data.dataloader.DataLoader: Pytorch dataloader.
        """
        return DataLoader(self,
                    shuffle=True,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    drop_last=True)

    def __len__(self) -> int:
        """
        Returns number of samples (blocks) of dataset

        Returns:
            length: int
        """
        return len(self.saved_hashes)

    def __del__(self) -> None:
        self.close()
        if hasattr(self, 'download_thread'):
            self.download_thread.join()

    def close(self) -> None:
        # Cancel sample tasks.
        if len(self.fetch_text_tasks)> 0:
            for t in self.fetch_text_tasks:
                t.cancel()

    
    @classmethod
    def test_dataset(cls):
        import commune
        # self = bittensor.dataset(batch_size=32, block_size=256)
        self = cls(batch_size=32, sequence_length=256, max_datasets=10, datasets=['ArXiv', 'Books3'])
        st.write(len(self))
        
        # print(self.download(chunk_size=200))
        # self.download()
        
        # t = commune.timer()

        # for i in range(100):
            
        #     x = next(self)
        #     print(f'Sample per Second: {i} {i/t.seconds}, SHAPE: {x["input_ids"].shape}')
            
    @classmethod
    def sandbox(cls):
        self = cls(batch_size=32, sequence_length=256, max_datasets=10, datasets=['ArXiv', 'Books3'])
        print('START')
        import commune
        
        print(len(self))
        # t = commune.timer()
        # for i in range(1000):
            # print(self.sample()['input_ids'].shape, i/t.seconds)
            
            
    @classmethod
    def deploy_swarm(cls):
        dataset_module = commune.get_module('dataset.text.bittensor')
        datasets = ['ArXiv', 'Gutenberg_PG', 'BookCorpus2', 'HackerNews', 'Books3', 'NIHExPorter', 'DMMathematics', 'OpenSubtitles']

        for dataset in datasets:
            module_name = f'dataset::{dataset.lower()}'
            cls.launch(name=module_name, kwargs={'datasets': dataset})
    @classmethod
    def test_swarm(cls):
        dataset_module = commune.get_module('dataset.text.bittensor')
        datasets = ['ArXiv', 'Gutenberg_PG', 'BookCorpus2', 'HackerNews', 'Books3', 'NIHExPorter', 'OpenSubtitles', 'DMMathematics']
    
        for dataset in datasets:
            module_name = f'dataset:{dataset.lower()}'
            
    
    @classmethod
    def sandbox(cls):
        import streamlit as st
        self = cls(batch_size=32, sequence_length=256, max_datasets=10, download=True)
        t = commune.timer()
        for i in range(100):
            
            print(self.sample()['input_ids'].shape,  int(i/t.seconds))
        

        # files = self.glob('saved_file_metas/*')
        # for i, h_url in enumerate(files):
        #     h = os.path.basename(h_url).split('.')[0]
        #     self.saved_hashes.append(h)
        #     if h in self.hash_dataset_map:
        #         dataset = self.hash_dataset_map[h]
        #         text = self.get_json(f'saved_file_metas/{h}.json')
        #         self.save_json(f'saved_file_metas/{dataset}/{h}.json', text )
        #         self.rm_json(f'saved_file_metas/{h}.json')
        #         if i %  100 == 0:
        #             st.write(f'{i}/{len(files)}')
        # for h in self.saved_hashes:
            
            
        # st.write(len(dataset))
        
    @classmethod
    def test(cls, *args,**kwargs):
        self = cls( *args,**kwargs)
        cls.print(self.sample())
        

        
        
if __name__ == "__main__":
    BittensorDataset.run()