
#!/usr/bin/env python3
##################
##### Import #####
##################
import sys, os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import ray
import torch
import concurrent.futures
import time
import psutil
import random
import argparse
from tqdm import tqdm
import asyncio
import numpy as np
import sys
import pandas as pd
from typing import Optional, Any, Union, Dict, List
from torch.utils.data.dataloader import DataLoader
from copy import deepcopy
import threading
import time
import queue
from munch import Munch
import inspect
from glob import glob
from importlib import import_module
import shutil
import streamlit as st
import bittensor
import commune

def tensor_stats(x):
    return {'std': x.std().item(), 'mean': x.mean().item()}


class CortexTrainer(commune.Module):
    def __init__(self, 
                batch_size = 10,
                sequence_length = 16,
                include_input= False,
                num_endpoints = 20,
                num_workers = 1,
                subtensor: Union[Dict[str,Any], bittensor.subtensor]=None,
                metagraph: Union[Dict[str,Any], bittensor.metagraph]=None,
                dataset: Union[Dict[str,Any], bittensor.dataset, 'HuggingfaceDataset']=None, 
                tokenizer: Union[Dict[str,Any], bittensor.tokenizer]=None,
                wallet: Union[Dict[str,Any], bittensor.wallet] = None,
                receptor_pool: Union[Dict[str,Any], bittensor.receptor_pool] = None,
                load:bool = True,
                config=None):

        # Avoid deadlocks when usinng tokenizer.
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.__infinite_dataset_iterator = None
        self.batch_size = batch_size
        self.num_endpoints = num_endpoints
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.config = self.load_config(config=config)

        if load:
            self.subtensor = self.set_subtensor(subtensor)
            self.metagraph = self.set_metagraph(metagraph)
            self.wallet = self.set_wallet(wallet)
            self.receptor_pool =self.set_receptor_pool(receptor_pool=None)
           
            self.dataset = self.set_dataset(dataset)
            self.tokenizer = self.set_tokenizer(tokenizer)

        self.tasks = []
        self.sample_cache = []
        self.task_queue = asyncio.Queue()
        self.sync_the_async()

    @property
    def sample_probability(self):
        # sample inversly proportional
        max_sample_size =self.max_sample_size
        sample_size_array = self.sample_size_array
        pre_normalized = (max_sample_size - sample_size_array)**2 +  1e-10
        normalized = pre_normalized / np.sum(pre_normalized)

        return normalized


    
    @property
    def sampleidx2size(self):
        x = {k:len(v) for k,v in self.sampleidx2result.items()}
        return x

    @property
    def sample_size_array(self):
        return np.array(list(self.sampleidx2size.values()))
    @property
    def total_sample_size(self):
        return int(np.sum(self.sample_size_array))

    @property
    def max_sample_size(self):
        return int(np.max(self.sample_size_array))

    @property
    def min_sample_size(self):
        return int(np.min(self.sample_size_array))

    def set_tokenizer(self):
        tokenizer = self.launch(**self.config['tokenizer'])
        return tokenizer

    def set_receptor_pool(self, receptor_pool=None):
        rp_config = deepcopy(self.config['receptor_pool'])
        rp_config['kwargs']['wallet']=self.wallet

        if receptor_pool == None:
            receptor_pool = self.launch( **rp_config)  
        self.receptor_pool = receptor_pool

        return self.receptor_pool

    def set_dataset(self, dataset:'dataset.huggingface'=None):
        if dataset==None:
            dataset = commune.launch(**self.config['dataset'])
        self.dataset = dataset
        self.dataset_id = '-'.join([self.dataset.path, self.dataset.name, self.dataset.split])
        self.sample_ids = [f'{self.dataset_id}/{i}' for i in range(self.idx_bounds[0], self.idx_bounds[1]) ]
        self.sampleidx2result = {i:[] for i in self.sample_ids}

        
        return self.dataset

    def set_wallet(self,wallet:bittensor.wallet) -> bittensor.wallet:
        wallet = wallet if wallet else self.config.wallet
        if isinstance(wallet, dict):
            self.wallet = bittensor.wallet(**wallet)
        elif isinstance(wallet, bittensor.wallet):
            self.wallet = wallet
        else:
            raise NotImplemented(f'{type(wallet)} type of wallet is not available')
    
        return self.wallet

    def set_tokenizer(self, tokenizer:Optional[bittensor.tokenizer]=None) -> bittensor.tokenizer:
        if tokenizer == None:
            tokenizer = bittensor.tokenizer()
        self.tokenizer = tokenizer
        return tokenizer

    default_sync_delay = 100
    @property
    def sync_delay(self)-> int:
        '''
        The number of blocks you want to pass before syncing the metagraph.
        '''
        delay_threshold = self.config.get('delay_threshold', self.default_sync_delay)
        return delay_threshold
    
    def set_subtensor(self, subtensor=None):
        if subtensor == None:
            self.subtensor = bittensor.subtensor( config = self.config.bittensor )
        return self.subtensor

    @property
    def current_block(self):
        return self.subtensor.block
    
    @property
    def synced_block(self): 
        return self.metagraph.block.item()

    @property
    def sync_delay(self):
        return self.current_block - self.synced_block
    
    def get_receptors(self, n = 10,uids=None):
        if uids == None:
            uids = list(range(n))
        receptors = []
        for uid in uids:
            receptors += [bittensor.receptor( wallet = self.wallet, endpoint = self.metagraph.endpoint_objs[uid])]
        return receptors
    

    @property
    def endpoints(self):
        endpoints =self.metagraph.endpoint_objs
        return endpoints
    
    @property
    def uidsf(self):
        return list(map(lambda x: x.uid, self.endpoints))
        
    def get_random_endpoints(self, n = 10 ):
        endpoints =self.endpoints
        random_ids =  np.random.randint(0, len(endpoints), (n))
        return [endpoints[i] for i in random_ids]

    def get_endpoints(self, n=10, uids:list=[]):
        if len(uids) == 0:
            uids = list(range(n))
        endpoints =self.metagraph.endpoint_objs
        selected_endpoints = []
        for uid in uids:
            selected_endpoints += [endpoints[uid]]
        return selected_endpoints

    @staticmethod
    def str2synapse(synapse:str, *args, **kwargs):
        return getattr(bittensor.synapse, synapse)(*args, **kwargs)
    @property
    def available_synapses(self):
        return [f for f in dir(bittensor.synapse) if f.startswith('Text')]

    synapses = all_synapses=available_synapses
    @staticmethod
    def errorcode2name(code):
        code2name_map =  {k:f'{v}' for k,v in zip(bittensor.proto.ReturnCode.values(),bittensor.proto.ReturnCode.keys())}
        return code2name_map[code]

        
    async def async_receptor_pool_forward(self, endpoints, inputs, synapses , timeout, min_successes, chunks=5):
        endpoints_split_list = chunk(endpoints, num_chunks=chunks)
        kwargs_list = []
        for endpoints_split in endpoints_split_list:
            kwargs_list.append(dict(endpoints=endpoints_split, inputs=inputs, synapses=synapses , timeout=timeout))
        results_list = await asyncio.gather(*[self.receptor_pool.async_forward(**kwargs) for kwargs in kwargs_list])
       
        agg_results = [[],[], []]
        for results in results_list:
            for i,result in enumerate(results):
                agg_results[i].extend(result)

        return agg_results

    def resolve_synapse(self, synapse:str, *args,**kwarga):
        return getattr(bittensor.synapse, synapse)()

    @property
    def idx_bounds(self):
        idx_bounds  = self.config['idx_bounds']
        assert idx_bounds[0]>=0
        return idx_bounds
    
    metagraph_keys = ['trust', 'consensus','stake', 'incentive', 'dividends', 'emission']

    @staticmethod
    def weighted_coin(prob:float):
        return bool(random.uniform(0,1) > heads_prob)


    async def async_cache_sample(self, 
                                batch_size:Optional[int]=None, 
                                num_endpoints:Optional[int]=None, 
                                seqeunce_length:Optional[int]=None,
                                cache_path:Optional[str]=None, 
                                refresh_cache:bool=False,
                                device:Optional[str]=None
                                ):
        

        sample_dict = {}


        if cache_path and refresh_cache==False:
            try:
                sample_dict = self.load_sample(cache_path)
            except FileNotFoundError:
                pass

        if len(sample_dict) == 0:
            batch_size = batch_size if batch_size else self.batch_size
            num_endpoints = num_endpoints if num_endpoints else self.num_endpoints
            
            sample_tasks = [ self.async_get_cached_sample(num_endpoints=num_endpoints) for i in range(batch_size)]
            sample_batch_list = await asyncio.gather(*sample_tasks)
            sample_batch  = {k:[] for k in sample_batch_list[0].keys()}
            for sample in sample_batch_list:
                for k in sample.keys():
                    sample_batch[k].append(sample[k])

            sample_dict = {k:torch.stack(v) if isinstance(v[0], torch.Tensor ) else v for k,v in sample_batch.items()}

            if cache_path:
                self.save_sample(cache_path, sample_dict)
            
        # sample_dict = self.process_batch(sample_dict)
        # if device:
        #     sample_dict = {k:v.to(device) for k,v in sample_dict.items()}
        
        return sample_dict


    async def async_sample(self,
            sequence_length:int = 20,
            batch_size:int = 10,
            synapses: Union[list, str, 'bittensor.Synapse'] = 'TextCausalLMNext',
            include_input: bool = True,
            min_successes:int =30,
            timeout:int= 4,
            idx_list: Optional[List[int]] = None,
            num_endpoints:int = 10,
            chunks:int=1, 
            save:bool = True,
            load_cache_ratio: float = 0.5,
            save_cache_ratio: float = 0.5,
        ):

        
        if not isinstance(synapses, list):
            synapses = [synapses]

        # Convert synapse strings to objects.
        synapses = [self.resolve_synapse(s) for s in synapses]

        # resolve path
        if idx_list:
            batch_size = len(idx_list)
        else:
            idx_list = np.random.choice(np.arange(self.idx_bounds[0], self.idx_bounds[1]), batch_size, p=self.sample_probability).tolist()

        # Get Random Endpoints and convert into Tensor.
        endpoints = self.get_random_endpoints(num_endpoints)

        sample_kwargs = dict( batch_size=batch_size, idx_list = idx_list, sequence_length=sequence_length, tokenize= False)
        input_dict =  {
            'dataset': {**self.dataset.card , 'sample_kwargs': sample_kwargs},
        }

        # Sample Raw Text and Tokenize it. 
        raw_text = self.dataset.sample(**sample_kwargs)['text']
        token_batch = self.tokenizer(raw_text, max_length=sequence_length+1, truncation=True, padding="max_length", return_tensors="pt")["token_batch"]
        

        next_input_id = token_batch[:, -1:]
        token_batch = token_batch[:, :-1]
        inputs = [token_batch]*len(endpoints)

        if include_input:
            input_dict['token_batch'] = token_batch
            input_dict['next_input_id'] = next_input_id



        io_1 = psutil.net_io_counters()
        start_bytes_sent, start_bytes_recv = io_1.bytes_sent, io_1.bytes_recv

        with Timer() as t:
            
            all_synapses_results = await self.async_receptor_pool_forward(
                                endpoints=endpoints,
                                synapses= synapses,
                                timeout=timeout,
                                min_successes=min_successes,
                                inputs=  inputs,
                                chunks=chunks)

            elapsed_time = t.elapsed_time.total_seconds() 

        io_2 = psutil.net_io_counters()
        total_bytes_sent, total_bytes_recved = io_2.bytes_sent - start_bytes_sent, io_2.bytes_recv - start_bytes_recv
        all_synapses_results = {'tensor': all_synapses_results[0],
                                'code': all_synapses_results[1],
                                 'latency':  all_synapses_results[2],
                                 'endpoints': endpoints}
        
        results = {k:[] for k in all_synapses_results.keys()}
        success_indices = []
        num_endpoints = len(all_synapses_results['tensor'])
        for r_i in range(num_endpoints):
            if all([c == 1 for c in all_synapses_results['code'][r_i]]):
                for k,v in all_synapses_results.items():
                    results[k].append(v[r_i])

        endpoints = results.pop('endpoints')
        results['uid']  = list(map(lambda e:e.uid, endpoints))
        results['hotkey'] = list(map(lambda e:e.hotkey, endpoints))
        results['code'] = list(map(lambda x:x[0], results['code']))
        results['latency'] = list(map(lambda x:x[0], results['latency']))

        result_keys = list(results.keys())
        metrics_dict = {}
        metrics_dict['num_successes'] = len(results['tensor'])
        # if metrics_dict['num_successes'] > 0:
        #     return None

        metrics_dict['timeout'] = timeout
        metrics_dict['num_endpoints'] = num_endpoints
        metrics_dict['batch_size'] = batch_size
        metrics_dict['sequence_length'] = sequence_length
        metrics_dict['chunks'] = chunks
        metrics_dict['elapsed_time'] = elapsed_time
        metrics_dict['successes_per_second'] = metrics_dict['num_successes']/metrics_dict['elapsed_time'] 
        metrics_dict['time_over_timeout'] = elapsed_time - timeout
        metrics_dict['time_over_timeout_ratio'] = (elapsed_time - timeout)/(timeout + 1e-10)
        metrics_dict['upload_bytes_mb'] =total_bytes_sent / 1000
        metrics_dict['download_bytes_mb'] =total_bytes_recved / 1000
        metrics_dict['upload_rate_mb'] =metrics_dict['upload_bytes_mb']/elapsed_time 
        metrics_dict['download_rate_mb'] =metrics_dict['download_bytes_mb']/elapsed_time
        metrics_dict['success_rate'] = metrics_dict['num_successes']/metrics_dict['num_endpoints']
        metrics_dict['num_tokens'] = batch_size*sequence_length

        async_save_tasks = []
        metagraph_state_dict = self.metagraph.state_dict()
        for k in self.metagraph_keys:
            results[k] =  metagraph_state_dict[k][results['uid']].tolist()

        results = dict(metrics=metrics_dict, **input_dict, **results)

        if cache_path:
            self.save_sample(cache_path, results)

        if save:
            
            for response_i in range(len(results['tensor'])):
                for synapse_i, synapse in enumerate(synapses):
                    for batch_i in range(batch_size):
                        idx = idx_list[batch_i]

                        sample = {}
                        for k,v in results.items():
                            if k == 'tensor':
                                sample[k] = v[response_i][synapse_i][batch_i]
                            elif k == 'metrics':
                                pass
                            elif type(v) in [list]:
                                sample[k] = v[response_i]
                            else:
                                sample[k] = v

                        sample.update(deepcopy(input_dict))
                        sample['dataset'].pop('sample_kwargs')
                        sample['dataset']['sequence_length'] = sequence_length
                        sample['dataset']['idx'] = idx
                        sample['synapse'] = str(synapse)
                        if include_input:
                            sample['token_batch'] = token_batch[batch_i]
                            sample['next_input_id'] = next_input_id[batch_i]




                        path = f'{self.dataset_id}_seqlen-{sequence_length}_inputs-{include_input}/sample-{idx}/hotkey_{sample["hotkey"]}/synapse-{str(synapse)}'

                        async_save_tasks += [self.async_save_sample(path=path,data=sample)]
        
            if len(async_save_tasks)>0:
                await asyncio.gather(*async_save_tasks)
        return results
    async def async_load_torch(self,path) -> dict:
        sample = torch.load(path)
        return sample

    async def async_load_paths(self, paths:list, mode='torch'):
        load_fn = getattr(self, f'async_load_{mode}')
        return await asyncio.gather(*[load_fn(path) for path in paths])


    async def async_torch_save(self, path, data):
        path = self.resolve_path(path)
        path = torch.save(data, path)
        return path

    async_save_sample = async_torch_save
    async_load_sample = async_load_torch
            

    def sample_generator(self, num_endpoints=10, 
                        sequence_length=10,
                         batch_size=10,
                        num_batches=10,
                         max_tasks=10,
                         min_successes=10,
                         save=True,
                         *args, **kwargs):

        jobs = []
        kwargs.update(dict(sequence_length=sequence_length, 
                            batch_size=batch_size,
                            num_endpoints=num_endpoints, 
                            min_successes=min_successes,
                            save=save))

        metrics_dict = {}
        with Timer() as t:
            for i in range(num_batches):
                self.submit_job(fn=self.async_sample, max_tasks=max_tasks, *args, **kwargs)
        
            finished_results = []
            failed_results = []
            for i in range(num_batches):
                finished_result = self.get_sample(max_tasks=max_tasks)
                if finished_result != None:
                    finished_results.append(finished_result)
                    num_finished_results = len(finished_results) + len(failed_results)
                    responses = sum([len(fr['tensor'])  for fr in finished_results])
                    
                    rate = num_finished_results/ t.seconds
                    percent_finished = ((num_finished_results)/num_batches)*100
                    num_failed_results = len(failed_results)

                    st.write(f'finished_results: {num_finished_results}, failed_results: {num_failed_results} rates: {rate} percent_finished: {percent_finished}% response_rate: {responses/len(finished_results)} ')
                else:
                    failed_results += [finished_result]

        
            metrics_dict['seconds'] = t.seconds
            metrics_dict['successes'] = len(finished_results)
            metrics_dict['input'] = dict(num_batches=num_batches, num_endpoints=num_endpoints, max_tasks=max_tasks)
            metrics_dict['samples'] = sum([len(fr['tensor']) * batch_size for fr in finished_results])
            metrics_dict['successful_endpoints'] = sum([len(fr['tensor'])for fr in finished_results])
            metrics_dict['samples_per_batch'] = metrics_dict['samples']/num_batches
            metrics_dict['success_rate'] = metrics_dict['samples']/(num_endpoints*num_batches)
            metrics_dict['min_success_rate'] = metrics_dict['samples']/(min_successes*num_batches)
            metrics_dict['tokens'] = sum([len(fr['tensor'])*batch_size * sequence_length for fr in finished_results ])
            metrics_dict['elapsed_time'] = sum( [fr['metrics']['elapsed_time'] for fr in finished_results if 'elapsed_time' in fr['metrics']])/(len([fr for fr in finished_results if 'elapsed_time' in fr['metrics']]) + 1e-10)
        
        for k in list(metrics_dict.keys()):
            if k not in ['seconds', 'success_rate', 'samples_per_batch','input']:
                metrics_dict[f'{k}_per_second'] = metrics_dict[k] / metrics_dict['seconds']
        return metrics_dict

    async def async_submit_job(self, fn, max_tasks=10, *args, **kwargs):
        await self.task_queue.put(dict(fn=fn, args=args, kwargs=kwargs))

    async def async_get_sample(self, max_tasks=10):

        if len(self.sample_cache) > 0:
            return self.sample_cache.pop(0)

        # ensure there is enough space
        free_task_slots = max_tasks - len(self.tasks) 
        if free_task_slots>0:
            for i in range(free_task_slots):
                if self.task_queue.empty() == True:
                    continue
                task_meta = await self.task_queue.get()

                self.tasks.append(task_meta['fn'](*task_meta['args'], **task_meta['kwargs']))

        finished_tasks, tasks = await asyncio.wait(self.tasks, return_when=asyncio.FIRST_COMPLETED)
        finished_tasks, self.tasks = list(finished_tasks), list(tasks)

        for finished_task in finished_tasks:
            finished_result = finished_task.result()
            self.sample_cache.append(finished_result)

        assert len(self.sample_cache) > 0
        
        return self.sample_cache.pop(0)
        

    @classmethod
    def sync_the_async(cls, obj:'class' = None):
        '''
        Syncing the Async
        '''
        if obj == None:
            obj = cls

        for f in dir(obj):
            if 'async_' in f:
                setattr(obj, f.replace('async_',  ''), cls.sync_wrapper(getattr(obj, f)))


    @staticmethod
    def sync_wrapper(fn):
        def wrapper_fn(*args, **kwargs):
            return asyncio.run(fn(*args, **kwargs))
        return  wrapper_fn


    @property
    def __file__(self):
        module_path =  inspect.getmodule(self).__file__
        return module_path

    def load_config(self, config:Optional[Union[str, dict]]=None):
        if config == None:
            config = load_yaml(self.config_path())
        elif isinstance(config, str):
            config =  load_yaml(config)
        elif isinstance(config, dict):
            config = config
        
        config = Munch(config)

        # Add bittensor in there.
        config.bittensor =  bittensor.config()
        
        return config



    @classmethod
    def import_module(cls, import_path:str) -> 'Object':
        return import_module(import_path)

    @classmethod
    def import_object(cls, key:str)-> 'Object':
        module = '.'.join(key.split('.')[:-1])
        obj =  getattr(import_module(module), object_name)
        return obj

    @classmethod
    def launch(cls, module:str, fn:Optional[str]=None ,kwargs:dict={}, args:list=[]):
        module_class = cls.import_object(module)
        module_kwargs = {**kwargs}
        module_args = [*args]
        
        if fn == None:
            module_object =  module_class(*module_args,**module_kwargs)
        else:
            module_init_fn = getattr(module_class,fn)
            module_object =  module_init_fn(*module_args, **module_kwargs)
        return module_object
    
    @property
    def get_bittensor_config(self):
        return bittensor.config(parser = self.argparser())
    def argparser(self) -> 'argparse.ArgumentParser':
        parser = argparse.ArgumentParser( 
            description=f"Bittensor Speed Test ",
            usage="python3 speed.py <command args>",
            add_help=True
        )
        bittensor.wallet.add_args(parser)
        bittensor.logging.add_args(parser)
        bittensor.subtensor.add_args(parser)
        return parser

    @property
    def module_path(self):
        local_path = self.__file__.replace(os.getenv('PWD'), '')
        local_path, local_path_ext = os.path.splitext(local_path)
        module_path ='.'.join([local_path.replace('/','.')[1:], self.__class__.__name__])
        return module_path


    @property
    def tmp_dir_path(self) -> str:
        '''
        The temporary directory path for storing any artifacts.

        Returns:
            tmp_dir_pat (str)
        '''

        # Remove the PWD.

        tmp_dir_path =  '/tmp' + self.__file__.replace(os.getenv('PWD'), '')


        # Remove the extension.
        tmp_dir_path, tmp_dir_path_ext = os.path.splitext(tmp_dir_path)
        if not os.path.isdir(tmp_dir_path):
            os.makedirs(tmp_dir_path)
        return tmp_dir_path

    def resolve_path(self, path):
        if self.tmp_dir_path not in path:
            path = os.path.join(self.tmp_dir_path, path)
        path = ensure_path(path)
        return path

    async def async_save_json(self,path:str, data:Union[list, dict], use_tmp_dir:bool=True) -> 'NoneType':
        if use_tmp_dir:
            path = self.resolve_path(path)
        return await async_save_json(path, data)

    async def async_load_json(self,path:str, use_tmp_dir:bool=True, handle_error:bool=False) -> 'NoneType':
        if use_tmp_dir:
            path = self.resolve_path(path)

        try:
            data = await async_load_json(path)
        except FileNotFoundError as e:
            if handle_error:
                return None
            else:
                raise e
        
        return data

    def rm(self,path:str ,use_tmp_dir:bool=True) -> Union['NoneType', str]:
        
        if use_tmp_dir:
            path = os.path.join(self.tmp_dir_path, path)
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path):
                path = ensure_json_path(path, ensure_extension=True, ensure_directory=False)
                os.remove(path)
            else:
                raise NotImplementedError(path)
            
            return path
        else:
            return None
    def glob_paths(self, query:Optional[str]=None) -> List[str]: 
        sample_paths = [f for f in glob(self.tmp_dir_path+'/**', recursive=True) if os.path.isfile(f)]
        if isinstance(query, str):
            sample_paths = [f for f in sample_paths if query in f]
        return sample_paths
    
    def get_sample_paths(self,sequence_length=None, include_input=True, synapses:List[str]=None, refresh=False) -> List[str]: 

        if sequence_length == None:
            sequence_length = self.sequence_length
        if hasattr(self, '_sample_paths') and not refresh:
            pass
        else:
            self._sample_paths = [f for f in glob(self.tmp_dir_path+f'/{self.dataset_id}_seqlen-{sequence_length}_inputs-{include_input}/**/*', recursive=True) if os.path.isfile(f)]

            # filter out paths with synapse (one only)
            if synapses:
                return [f for f in self._sample_paths if all([s in f for s in synapses])]
            
            self._sample_paths = list(set(['/'.join(p.split('/')[:-2]) for p in self._sample_paths]))

        return self._sample_paths

    def list_paths(self, query):
        paths = [f for f in glob(f'{self.tmp_dir_path}/{query}/**', recursive=True) if os.path.isfile(f) ]
        return paths

    def get_sample_path(self, idx:int=None, refresh=False, sequence_length=None, include_input=True, synapses:List[str]=None):
        sample_paths = self.get_sample_paths(refresh=refresh, sequence_length=sequence_length, include_input=include_input, synapses=synapses)
        idx = idx if idx else random.randint(0, len(sample_paths)-1)
        assert idx >= 0 and idx < len(sample_paths), idx
        return sample_paths[idx]

    def get_sample_response_paths(self, sample_path:str, synapses:List[str]=None) -> List[str]:
        assert os.path.isdir(sample_path)
        response_paths = glob(sample_path+'/*')
        assert len(response_paths) > 0, f'{sample_path} is empty'
        if synapses:
            # filter paths with all of the synapses in them still
            response_paths =   [p for p in response_paths if all([s in p for s in synapses])]
        return response_paths
    async def async_get_cached_sample(self, idx:int = None, num_endpoints = 10, max_trials=10):
        sample_response_paths  = []
        while len(sample_response_paths) < num_endpoints:
            sample_path = self.get_sample_path(idx=idx)
            sample_response_paths = self.get_sample_response_paths(sample_path)
        sample_response_paths = sample_response_paths[:num_endpoints]
        sample_response_tasks = [self.async_get_cached_response(p) for p in sample_response_paths]
        sample_responses = await asyncio.gather(*sample_response_tasks)

        return sample

    async def async_get_cached_response(self, response_path):
        synapse_paths = glob(response_path+'/**')
        synapse_samples = await asyncio.gather(*[ self.async_load_sample(synapse_path) for synapse_path in synapse_paths ])

        sample = {}
        for synapse_i,synapse_path in enumerate(synapse_paths):
            synapse = os.path.splitext(synapse_path)[0].split('synapse-')[-1]
            if len(sample) == 0:
                sample = synapse_samples[synapse_i]
                sample['tensor'] = {synapse:sample.pop('tensor')}
            else:
                sample['tensor'][synapse] = synapse_samples[synapse_i]['tensor']

        synapses = list(sample['tensor'].keys())
        for synapse in synapses:
            sample[synapse] = sample['tensor'][synapse]


        for k in [ 'code', 'synapse', 'uid', 'latency', 'tensor']:
            sample.pop(k, None)


        for k,v in sample.items():
            if type(v) in [int, float, list]:
                sample[k] = torch.tensor(v)



        sample['synapses'] = synapses

        return sample
    def __getitem__(self, idx:int=None):
        return self.get_cached_sample(idx=idx)

    def __len__(self):
        return len(self.get_sample_paths())



    def process_batch(self, x):
        processed_batch = {}
        endpoint_emb = []
        for k in self.metagraph_keys:
            k_mean = self.metagraph_statistics[k]['mean']
            k_std = self.metagraph_statistics[k]['std']
            v = (torch.tensor(x[k])-k_mean)/k_std
            endpoint_emb.append(v)
            processed_batch[k] = x[k]
            

        endpoint_emb = torch.stack(endpoint_emb,dim=-1)
        processed_batch['endpoint_emb'] = endpoint_emb

        sequence_length = x['TextCausalLMNext'].shape[-2]
        processed_batch['prediction'] = x['TextCausalLMNext']
        processed_batch['gt'] = x['next_input_id']
        processed_batch = {k:v.transpose(0,1) for k,v in processed_batch.items()}
        processed_batch['gt'] = processed_batch['gt'][0]


        return processed_batch

    def dataloader(self) -> DataLoader:
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
    
    def __next__(self) -> Dict[str, torch.tensor]:
        """
        Returns the next element from the dataset. 
        """
        if self.__infinite_dataset_iterator == None:
            self.__infinite_dataset_iterator = iter(self.dataloader())

        try:
            return next(self.__infinite_dataset_iterator)
        except StopIteration:
            self.__infinite_dataset_iterator = iter(list(self.dataloader()))
            return next(self.__infinite_dataset_iterator)



    @staticmethod
    def get_sample_schema(x: Dict[str, torch.Tensor]) -> Dict[str, dict]:
        # Get the schema of the 
        assert isinstance(x, dict), 'x must be a dictionary'
        sample_schema = {}
        for k,v in x.items():
            if isinstance(v, torch.Tensor):
                sample_schema[k] = dict(type='torch',shape=v.shape, dtype=v.dtype, device=v.device)
            elif isinstance(v, list):
                sample_schema[k] = dict(type='list', length=len(v), dtypes=[type(e) for e in v])
            else:
                sample_schema[k] = dict(type=str(type(v)))

        return sample_schema

    def set_metagraph(self, metagraph=None, delay_threshold:int=None):
        if metagraph == None:
            self.metagraph = bittensor.metagraph( subtensor = self.subtensor )
        else:
            self.metagraph = metagraph
        self.metagraph.load()
        self.sync_metagraph(delay_threshold=delay_threshold)
    
        return self.metagraph

    @property
    def max_block_delay(self):
        metagraph_block_delay_threshold = self.config.get('delay_threshold', 100)

    @property
    def should_update_metagraph(self, delay_threshold:int=100):
        delay_threshold = delay_threshold if delay_threshold else self.max_block_delay
        return bool(self.sync_delay > delay_threshold)

    def sync_metagraph(self, delay_threshold:int=None):
        delay_threshold =  delay_threshold if delay_threshold else self.config.get('delay_threshold', 100)
        if self.sync_delay > self.config.get('delay_threshold', 100):
            self.metagraph.sync()
            self.metagraph.save()
        self.metagraph_state_dict = self.metagraph.state_dict()

        self.metagraph_statistics = {}
        for k in self.metagraph_keys:
            v = self.metagraph_state_dict[k]
            self.metagraph_statistics[k] = {'mean': v.mean().item(), 'std': v.std().item()}
        return self.metagraph

    @classmethod
    def test_speed(cls, steps=100, batch_size = 32, warmup_steps = 10):
        module = cls(batch_size=batch_size)
        for i in range(steps):
            if i >= warmup_steps:
                if i == warmup_steps:
                    t = Timer()
                    t.__enter__()
                st.write((i-warmup_steps)/t.seconds)
            x = module.get_batch()

    def load_experiment(self, results_path='experiment'):
        results_path = os.path.join('experiment',results_path )
        row_paths = self.list_paths(results_path)
        loaded_rows = self.load_paths(row_paths)
        return pd.DataFrame(loaded_rows)

    def set_model(self,model = None, device='cpu'):
        model = model if model else BaseMOE(input_dim = len(self.metagraph_keys))
        self.model = model
        return self.model


    def run_experiment(self, 
                        num_endpoints:int = 32,
                        batch_size:int = 16,
                        sample_cache_path:str = 'demo',
                        sample_refresh_cache:bool = False,
                        results_path:str = 'experiment',
                        refresh_results:bool = True,
                        num_batches:int = 1000,
                        add_metagraph_keys = True, 
                        response_df = {}):

        if results_path:
            results_path = os.path.join('experiment',results_path )
        if refresh_results:
            self.rm_json(results_path)

        input_row_dict = {'num_endpoints': num_endpoints, 'batch_size': batch_size}
        df = []
        self.model = self.set_model(device='cuda')
        for i in range(num_batches):
            x = self.get_batch(num_endpoints=num_endpoints, batch_size=batch_size, cache_path=f'{sample_cache_path}/sample_{i}', refresh_cache=sample_refresh_cache)
            
            st.write(x['prediction'].shape)
            # self.model.learn_step(x)

            gt = x['gt'].unsqueeze(0).repeat(num_endpoints, 1,1)
            per_response_loss = []
            model_ids = []
            individual_preds = x['prediction']
            for n_i in range(num_endpoints):
                per_response_loss.append(phrase_cross_entropy(target_phrases=gt[n_i],topk_tensor=individual_preds[n_i], reduce=True)[0])
                model_ids.append(f'model_{n_i}')
            per_response_loss = torch.stack(per_response_loss)
            sigma_distance = (per_response_loss - per_response_loss.mean())/ per_response_loss.std()
            row_dict = deepcopy(input_row_dict)
            row_dict['sigma_distance'] = sigma_distance.tolist()
            row_dict['loss'] = per_response_loss.tolist()
            if add_metagraph_keys:
                for metagraph_key in self.metagraph_keys:
                    row_dict[metagraph_key] = x[metagraph_key][:,0].tolist()
            
            utc_timestamp = time.time()
            if results_path:
                self.torch_save(path=f'{results_path}/row_{utc_timestamp}' , data=row_dict)
            df.append(row_dict)

    def plot_experiments(self, df):
        df=self.load_experiment()
        st.write(df)
        response_df  = {}
        for row_dict in df.to_dict(orient="records"):
            for k,v in row_dict.items():
                if isinstance(v, list):
                    if k in response_df:
                        response_df[k].extend(v)
                    else:
                        response_df[k] = v
        
        response_df = pd.DataFrame(response_df)
        from commune.streamlit import StreamlitModule, row_column_bundles
        StreamlitModule().run(response_df)

if __name__ == '__main__':

    # c.set_page_config(layout="wide")
    self = CortexTrainer(load=True, batch_size = 32, sequence_length=16, num_workers=1)
    self.cache_sample()
    # st.write(self.sample(synapses='TextCausalLMNext', timeout=4))
