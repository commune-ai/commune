
##################
##### Import #####
##################
import ray
import torch
import concurrent.futures
import time
import psutil
import random
import argparse
from tqdm import tqdm
import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
import bittensor
import streamlit as st
import numpy as np
import sys
import pandas as pd

from commune.utils import chunk
from copy import deepcopy
##########################
##### Get args ###########
##########################
from commune.streamlit import StreamlitPlotModule, row_column_bundles

import threading
import time
import queue
from loguru import logger

from commune import Module


parser = argparse.ArgumentParser( 
    description=f"Bittensor Speed Test ",
    usage="python3 speed.py <command args>",
    add_help=True
)
bittensor.wallet.add_args(parser)
bittensor.logging.add_args(parser)
bittensor.subtensor.add_args(parser)
config = bittensor.config(parser = parser)

class Sandbox(Module):
    sample_example = {}
    def __init__(self, 
                subtensor=None,
                dataset=None, 
                tokenizer=None,
                wallet = None,
                config=None, 
                loop=None):
        Module.__init__(self, config=config)

        self.sample_example = {}
        self.subtensor = self.set_subtensor(subtensor)
        self.wallet = self.set_wallet(wallet)
        self.receptor_pool =self.set_receptor_pool(receptor_pool=None)
        self.dataset = self.set_dataset(dataset)
        self.tokenizer = self.set_tokenizer(tokenizer)
            
        self.dataset_id = '.'.join([self.dataset.path, self.dataset.name, self.split])

        self.sample_ids = [f'{self.dataset_id}.{i}' for i in range(self.idx_bounds[0], self.idx_bounds[1]) ]
        
        self.sampleidx2result = {i:[] for i in self.sample_ids}
        
        
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

        st.write(normalized)

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



    @property
    def split(self):
        if 'split' not in self.config:
            self.config['split'] = 'train'
        return self.config['split']

    def set_tokenizer(self):
        tokenizer = self.launch(**self.config['tokenizer'])
        return tokenizer

    def set_receptor_pool(self, receptor_pool=None, refresh=None, max_active_receptors=0):
        rp_config = self.config['receptor_pool']
        rp_config['actor'] = rp_config.get('actor', False)
        rp_config['kwargs']['wallet']=self.wallet
        rp_config['kwargs']['max_active_receptors'] = max_active_receptors
        rp_config['kwargs']['compression'] = None

        if receptor_pool == None:
            receptor_pool = self.launch_module( **rp_config)  
        self.receptor_pool = receptor_pool

        return self.receptor_pool

    def set_dataset(self, dataset=None):
        if dataset==None:
            dataset = self.launch_module(**self.config['dataset'])
        
        self.dataset = dataset
        return self.dataset

    def set_wallet(self, wallet=None):
        wallet = wallet if wallet else self.config.get('wallet')
        if isinstance(wallet, dict):
            self.wallet = bittensor.wallet(**wallet)
        elif isinstance(wallet, bittensor.wallet):
            self.wallet = wallet
        else:
            raise NotImplemented(f'{type(wallet)} type of wallet is not available')
    
        return self.wallet

    def set_tokenizer(self, tokenizer=None):
        if tokenizer == None:
            tokenizer = bittensor.tokenizer()
        self.tokenizer = tokenizer
        return tokenizer
    
    def set_subtensor(self, subtensor=None):

        if subtensor == None:
            subtensor = bittensor.subtensor( config = config )
            graph = bittensor.metagraph( subtensor = subtensor )
            graph.load()
            self.subtensor = subtensor
            self.graph = graph
        if self.sync_delay > self.config.get('delay_threshold', 100):
            self.graph.sync()
            self.graph.save()

        return self.subtensor
    
    @property
    def current_block(self):
        return self.subtensor.block
    
    @property
    def synced_block(self): 
        return self.graph.block.item()

    @property
    def sync_delay(self):
        return self.current_block - self.synced_block
    
    def get_receptors(self, n = 10,uids=None):
        if uids == None:
            uids = list(range(n))
        receptors = []
        for uid in uids:
            receptors += [bittensor.receptor( wallet = self.wallet, endpoint = self.graph.endpoint_objs[uid])]
        return receptors
    

    @property
    def endpoints(self):
        endpoints =self.graph.endpoint_objs
        return endpoints
    
    @property
    def uids(self):
        return list(map(lambda x: x.uid, self.endpoints))
        
    def get_random_endpoints(self, n = 10 ):
        endpoints =self.endpoints
        random_ids =  np.random.randint(0, len(endpoints), (n))
        return [endpoints[i] for i in random_ids]

    def get_endpoints(self, n=10, uids:list=[]):
        if len(uids) == 0:
            uids = list(range(n))
        endpoints =self.graph.endpoint_objs
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

        
    async def async_receptor_pool_forward(self, endpoints, inputs, synapses , timeout, min_successes, splits=5):
        endpoints_split_list = self.chunk(endpoints, num_chunks=splits)
        kwargs_list = []
        for endpoints_split in endpoints_split_list:
            kwargs_list.append(dict(endpoints=endpoints_split, inputs=inputs, synapses=synapses , timeout=timeout, min_successes=min_successes))
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

    async def async_sample(self,
            sequence_length = 20,
            batch_size = 10,
            min_successes=30,
            timeout= 2,
            synapse = 'TextCausalLMNext',
            num_endpoints = 100,
            success_only= True,
            split = 'train', 
            splits=1, 
        ):

        idx_list = np.random.choice(np.arange(self.idx_bounds[0], self.idx_bounds[1]), batch_size, p=self.sample_probability).tolist()
        

        # inputs = torch.zeros([batch_size, sequence_length], dtype=torch.int64)
        raw_inputs = self.dataset.sample( batch_size=batch_size, idx_list = idx_list, split=split, sequence_length=sequence_length, tokenize= False)['text']
        inputs = self.tokenizer(raw_inputs, max_length=sequence_length, truncation=True, padding="max_length", return_tensors="pt")["input_ids"]
        synapse_str = deepcopy(synapse)   
        synapse = self.resolve_synapse(synapse)
        endpoints = self.get_random_endpoints(num_endpoints)
        uids = torch.tensor([e.uid for e in endpoints])
        input_dict =  {
            'split': split,
            'idx_list': idx_list,
            'dataset': self.dataset.config.get('dataset'),
            'raw_text': raw_inputs, 'input_ids': inputs,
            'synapse': synapse_str,
            'uids': uids
        }


        io_1 = psutil.net_io_counters()
        start_bytes_sent, start_bytes_recv = io_1.bytes_sent, io_1.bytes_recv

        with self.timer() as t:
            
            results = await self.async_receptor_pool_forward(
                                endpoints=endpoints,
                                synapses= [synapse],
                                timeout=timeout,
                                min_successes=min_successes,
                                inputs= [inputs]*len(endpoints),
                                splits=splits)

            elapsed_time = t.elapsed_time.total_seconds() 

        io_2 = psutil.net_io_counters()
        total_bytes_sent, total_bytes_recved = io_2.bytes_sent - start_bytes_sent, io_2.bytes_recv - start_bytes_recv

        results = list(results) + [list(map(lambda e:e.uid, endpoints))]
        results = self.process_results(results)


        success_indices = torch.argwhere(results['code']==1).squeeze(1).tolist()
        metrics_dict = {}
        metrics_dict['elapsed_time'] = elapsed_time
        metrics_dict['timeout'] = timeout
        metrics_dict['num_successes'] = len(success_indices)
        metrics_dict['successes_per_second'] = metrics_dict['num_successes']/metrics_dict['elapsed_time'] 
        metrics_dict['time_over_timeout'] = elapsed_time - timeout
        metrics_dict['time_over_timeout_ratio'] = (elapsed_time - timeout)/(timeout + 1e-10)
        metrics_dict['upload_bytes_mb'] =total_bytes_sent / 1000
        metrics_dict['download_bytes_mb'] =total_bytes_recved / 1000
        metrics_dict['upload_rate_mb'] =metrics_dict['upload_bytes_mb']/elapsed_time 
        metrics_dict['download_rate_mb'] =metrics_dict['download_bytes_mb']/elapsed_time
        metrics_dict['num_endpoints'] = num_endpoints
        metrics_dict['success_rate'] = metrics_dict['num_successes']/metrics_dict['num_endpoints']
        metrics_dict['splits'] = splits
        # results['output_size'] = sys.getsizeof( results.pop['tensor'])
        metrics_dict['batch_size'] = batch_size
        metrics_dict['sequence_length'] = sequence_length
        metrics_dict['num_tokens'] = batch_size*sequence_length

        if success_only and len(success_indices) == 0:
            return {}

        result_keys = ['tensor', 'code', 'uid']

        # results['code'] = list(map(self.errorcode2name, results['code'].tolist()))


        graph_state_dict = self.graph.state_dict()
        graph_keys = ['trust', 'consensus','stake', 'incentive', 'dividends', 'emission']
        for k in graph_keys:
            results[k] =  graph_state_dict[k][results['uid']]
        

        if success_only:
            for k in result_keys + graph_keys:
                if len(success_indices)>0:
                    results[k] = results[k][success_indices]
                else:
                    results[k] = []

        
        self.sample_example = results


        output = dict(info=metrics_dict, output=results, input=input_dict)

        singleton_result_dict = {}
        for k,v in output['output'].items():
            
            if len(v.shape) == 1:
                for endpoint_batch_tensor in torch.split(v, split_size_or_sections=1, dim=0 ):
                    endpoint_batch_tensor = endpoint_batch_tensor.squeeze(0).item()
                    if k in singleton_result_dict:
                        singleton_result_dict[k].append( endpoint_batch_tensor)
                    else:
                        singleton_result_dict[k] = [ endpoint_batch_tensor]
        single_results_dict = {}

        for k,v in output['output'].items():
            if len(v.shape)>1:
                for endpoint_batch_tensor in torch.split(v, split_size_or_sections=1, dim=1 ):
                    endpoint_batch_tensor = endpoint_batch_tensor.squeeze(1)
                    endpoint_batch_list = torch.split(endpoint_batch_tensor, split_size_or_sections=1, dim=0 )
                    if k in single_results_dict:
                        single_results_dict[k].append( endpoint_batch_list)
                    else:
                        single_results_dict[k] = [endpoint_batch_list]

        single_results_list = []

        for i in range(len(list(single_results_dict.values())[0])):
            row_dict = {**{k:v[i] for k,v in single_results_dict.items()} , **singleton_result_dict}
            single_results_list.append(row_dict)
            

        for i in range(len(idx_list)):
            idx = idx_list[i]
            for e in range(len(list(single_results_list[i].values())[0])):
                key = f'{self.dataset_id}.{idx}'
                self.sampleidx2result[key] += [{k:v[e] for k,v in single_results_list[i].items()}]
        return output

    def sample_generator(self, num_endpoints=10, 
                        sequence_length=10,
                         batch_size=10,
                        num_batches=10,
                         max_tasks=10,
                         min_successes=10,
                         *args, **kwargs):

        jobs = []
        kwargs.update(dict(sequence_length=sequence_length, batch_size=batch_size, num_endpoints=num_endpoints, min_successes=min_successes))

        metrics_dict = {}
        with self.timer() as t:
            for i in range(num_batches):
                self.submit_job(fn=self.async_sample, max_tasks=max_tasks, *args, **kwargs)
        
            finished_results = []
            for i in range(num_batches):
                finished_results.append(self.get_sample(max_tasks=max_tasks))

        
            metrics_dict['seconds'] = t.seconds
            metrics_dict['successes'] = len(finished_results)
            metrics_dict['input'] = dict(num_batches=num_batches, num_endpoints=num_endpoints, max_tasks=max_tasks)
        
            metrics_dict['samples'] = sum([fr['tensor'].shape[0] * fr['tensor'].shape[1] for fr in finished_results if 'tensor' in fr])
            metrics_dict['successful_endpoints'] = sum([fr['tensor'].shape[0] for fr in finished_results if 'tensor' in fr])


            metrics_dict['samples_per_batch'] = metrics_dict['samples']/num_batches
            metrics_dict['success_rate'] = metrics_dict['samples']/(num_endpoints*num_batches)
            metrics_dict['min_success_rate'] = metrics_dict['samples']/(min_successes*num_batches)
            metrics_dict['tokens'] = sum([fr['tensor'].shape[0]*fr['tensor'].shape[1] * sequence_length for fr in finished_results if 'tensor' in fr])
            metrics_dict['elapsed_time'] = sum( [fr['elapsed_time'] for fr in finished_results if 'elapsed_time' in fr])/len([fr for fr in finished_results if 'elapsed_time' in fr])
        
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
            self.sample_cache.append(finished_task.result())

        assert len(self.sample_cache) > 0
        return self.sample_cache.pop(0)
        

    def process_results(self, results):
        results_dict = {'tensor':[], 'code':[], 'uid': []}

        num_responses = len(results[0])
        for i in range(num_responses):
            tensor = results[0][i][0]
            code = results[1][i][0]
            latency = results[2][i][0]
            endpoint = results[3][i]

            results_dict['tensor'].append(tensor[...,:2])
            results_dict['code'].append(code)
            results_dict['uid'].append(endpoint)

        if len(results_dict['tensor'])>0:
            results_dict['tensor'] = torch.stack(results_dict['tensor'])
            results_dict['code'] = torch.tensor(results_dict['code'])
            results_dict['uid'] = torch.tensor(results_dict['uid'])
        else:
            results_dict['tensor'] = torch.tensor([])
            results_dict['code'] = torch.tensor([])
            results_dict['uid'] =  torch.tensor([])

        return results_dict

    def run_experiment(self,
            params = dict(
                sequence_length=[16,32,64, 128, 256 ],
                batch_size=[4,8,16,32, 64],
                num_endpoints=[32,64,128, 256],
                timeout=[4,8,12],
                synapse=['TextCausalLMNext'],
                splits=[1,2,4,8]
            ),
            experiment='experiment3',
            sequence_length=[]):
            
        sample_kwargs_list = []
        for sequence_length in params['sequence_length']:
            for num_endpoints in params['num_endpoints']:
                for timeout in params['timeout']:
                    for synapse in params['synapse']:
                        for batch_size in params['batch_size']:
                            for splits in params['splits']:
                                sample_kwargs_list += [dict(
                                    sequence_length = sequence_length,
                                    batch_size = batch_size,
                                    timeout= timeout,
                                    synapse = synapse,
                                    num_endpoints = num_endpoints,
                                    success_only= False,
                                    splits=splits
                                )]
        random.shuffle(sample_kwargs_list)
        for i,sample_kwargs in enumerate(tqdm(sample_kwargs_list)):
            self.set_receptor_pool(refresh=True)         
            trial_metrics_result = self.sample(**sample_kwargs)
            self.put_json(f'{experiment}/{i}', trial_metrics_result)
  
    # def streamlit(self):
    #     for k,v_list in params.items():
    def streamlit(self):
        st.write(self.load_experiment('experiment3'))


    def load_experiment(self, path='experiment3'):
        df = []
        
        for p in self.glob_json(path+'/*'):
            df.append(self.client.local.get_json(p))

        df =  pd.DataFrame(df)

        return df

    def streamlit_experiment(self, experiment= 'experiment3'):
        df = self.load_experiment(path=experiment)
        from commune.streamlit import StreamlitPlotModule, row_column_bundles

        df['tokens_per_second'] = df['num_tokens']*df['num_successes'] / df['elapsed_time']
        df['samples_per_second'] = df['batch_size']*df['num_successes'] / df['elapsed_time']
    
        StreamlitPlotModule().run(df)



    @staticmethod
    def chunk(sequence,
            chunk_size=None,
            append_remainder=False,
            distribute_remainder=True,
            num_chunks= None):
        # Chunks of 1000 documents at a time.

        if chunk_size is None:
            assert (type(num_chunks) == int)
            chunk_size = len(sequence) // num_chunks

        if chunk_size >= len(sequence):
            return [sequence]
        remainder_chunk_len = len(sequence) % chunk_size
        remainder_chunk = sequence[:remainder_chunk_len]
        sequence = sequence[remainder_chunk_len:]
        sequence_chunks = [sequence[j:j + chunk_size] for j in range(0, len(sequence), chunk_size)]

        if append_remainder:
            # append the remainder to the sequence
            sequence_chunks.append(remainder_chunk)
        else:
            if distribute_remainder:
                # distributes teh remainder round robin to each of the chunks
                for i, remainder_val in enumerate(remainder_chunk):
                    chunk_idx = i % len(sequence_chunks)
                    sequence_chunks[chunk_idx].append(remainder_val)

        return sequence_chunks

    def schema(self):
          return {k:v.shape for k,v in self.sample_example.items()}

    @classmethod
    def sync_the_async(cls, obj = None):
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

from munch import Munch 
class AyncioManager:
    """ Base threadpool executor with a priority queue 
    """

    def __init__(self,  max_tasks:int=10):
        """Initializes a new ThreadPoolExecutor instance.
        Args:
            max_threads: 
                The maximum number of threads that can be used to
                execute the given calls.
        """
        self.max_tasks = max_tasks
        self.running, self.stopped = False, False
        self.tasks = []
        self.queue = Munch({'in':queue.Queue(), 'out':queue.Queue()})
        self.start()

    def stop(self):
        while self.running:
            self.stopped = True
        return self.stopped
        
    def start(self):
        self.background_thread = threading.Thread(target=self.run_loop, args={}, kwargs={}, daemon=True)
        self.background_thread.start()

    def run_loop(self):
        return asyncio.run(self.async_run_loop())
    def new_aysnc_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
    async def async_run_loop(self): 
        loop = self.new_aysnc_loop()
        print(loop)
        self.stopped = False
        self.running = True
        print(loop)

        while self.running and not self.stopped:
            finished_tasks = []
            if len(self.tasks)>0:
                finished_tasks, self.tasks = await asyncio.wait(self.tasks)
            for finished_task in finished_tasks:
                self.queue.out.put(await asyncio.gather(*finished_task))
            if len(self.tasks) <= self.max_tasks:
                new_job = self.queue['in'].get()
                self.submit(**new_job)
                new_job = self.queue.out.get()

        loop.close()
        self.running = False

    def submit(self,fn, *args, **kwargs):
        job = {'fn': fn, 'args': args, 'kwargs': kwargs}
        self.queue['in'].put(job)

    def get(self):
        return self.queue['out'].get()

    def close(self):
        for task in self.tasks:
            task.cancel()
        self.stop()
        self.background_thread.join()

    def __del__(self):
        self.close()

if __name__ == '__main__':
    # Sandbox.ray_start()
    module = Sandbox.deploy(actor=False)
    for i in range(10):
        module.sample(batch_size=128, num_endpoints=10, timeout=4)


        metrics = dict(
                total_bin_size = module.total_sample_size,
                min_bin_size = module.min_sample_size, 
                max_bin_size = module.max_sample_size,
                num_samples = len(module.sampleidx2result)
                
                )
        metrics['mean_bin_size'] = metrics['total_bin_size'] / metrics['num_samples']

        st.write(metrics)

    

