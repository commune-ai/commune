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
import bittensor
import streamlit as st
import numpy as np
import sys
import pandas as pd
import nest_asyncio

##########################
##### Get args ###########
##########################
from commune.streamlit import StreamlitPlotModule, row_column_bundles

parser = argparse.ArgumentParser( 
    description=f"Bittensor Speed Test ",
    usage="python3 speed.py <command args>",
    add_help=True
)
bittensor.wallet.add_args(parser)
bittensor.logging.add_args(parser)
bittensor.subtensor.add_args(parser)
config = bittensor.config(parser = parser)
config.wallet.name = 'const'
config.wallet.hotkey = 'Tiberius'
##########################
##### Setup objects ######
##########################
# Sync graph and load power wallet.


from commune import Module
class Sandbox(Module):
    def __init__(self, 
                subtensor=None,
                dataset=None, 
                tokenizer=None,
                wallet = None,
                config=None, 
                load=False):
        Module.__init__(self, config=config)
        nest_asyncio.apply()
        # config = bittensor.config()
        if load:
            self.subtensor = self.set_subtensor(subtensor)
            self.wallet = self.set_wallet(wallet)
            self.receptor_pool =self.set_receptor_pool(receptor_pool=None)
            self.dataset = self.set_dataset(dataset)
            self.tokenizer = self.set_tokenizer(tokenizer)
    
    def set_receptor_pool(self, receptor_pool=None, refresh=None, max_active_receptors=0):
        rp_config = self.config['receptor_pool']
        # if refresh:
        #     rp_config['actor'] =  rp_config.get('actor',{})
        #     rp_config['actor']['refresh'] = True
        rp_config['actor'] = rp_config.get('actor')
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
        if wallet == None:
            wallet = bittensor.wallet(**self.config.get('wallet'))

        self.wallet = wallet
        return self.wallet

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
    

    def get_random_endpoints(self, n = 10 ):
        endpoints =self.graph.endpoint_objs
        random_ids =  list(np.random.randint(0, len(endpoints), (n)))
        return [endpoints[i] for i in random_ids]

    def get_endpoints(self, n=10, uids:list=[]):

        if len(uids) == 0:
            uids = list(range(n))
        endpoints =self.graph.endpoint_objs
        selected_endpoints = []
        for uid in uids:
            selected_endpoints += [endpoints[uid]]

        return selected_endpoints


    def tokenize(self, text:str, dtype = torch.int64, device='cpu'):
        # must be a string, or a list of strings
        if isinstance(text, str):
            text = [text]
        assert all(isinstance(t, str) for t in text)
        token_ids =  self.tokenizer(text)['input_ids']
        token_ids = torch.Tensor(token_ids).type(dtype).to(device)
        return token_ids

    @staticmethod
    def str2synapse(synapse:str, *args, **kwargs):
        return getattr(bittensor.synapse, synapse)(*args, **kwargs)
    @property
    def available_synapses(self):
        return [f for f in dir(bittensor.synapse) if f.startswith('Text')]


    @staticmethod
    def errorcode2name(code):
        code2name_map =  {k:f'{v}' for k,v in zip(bittensor.proto.ReturnCode.values(),bittensor.proto.ReturnCode.keys())}
        return code2name_map[code]

    def sample(self,
            sequence_length = 10,
            batch_size = 10,
            timeout= 4,
            synapse = 'TextLastHiddenState',
            num_endpoints = 30,
            success_only= True,
            return_type='results'
        ):
        # inputs = torch.zeros([batch_size, sequence_length], dtype=torch.int64)
        inputs = self.dataset.sample( batch_size=batch_size, sequence_length=sequence_length)

        synapse = getattr(bittensor.synapse, synapse)()
        endpoints = self.get_random_endpoints(num_endpoints)
        
        uids = torch.tensor([e.uid for e in endpoints])

        io_1 = psutil.net_io_counters()
        start_bytes_sent, start_bytes_recv = io_1.bytes_sent, io_1.bytes_recv

        with self.timer(text='Querying Endpoints: {t}', streamlit=True) as t:
            
            results = self.receptor_pool.forward(
                                endpoints=endpoints,
                                synapses= [synapse],
                                timeout=timeout,
                                inputs= [inputs]*len(endpoints))
            elapsed_time = t.elapsed_time.total_seconds() 

        io_2 = psutil.net_io_counters()
        total_bytes_sent, total_bytes_recved = io_2.bytes_sent - start_bytes_sent, io_2.bytes_recv - start_bytes_recv

        results = list(results) + [list(map(lambda e:e.uid, endpoints))]
        results = self.process_results(results)
        # tensors =
        
        
        success_indices = torch.argwhere(results['code']==1).squeeze(1).tolist()
        
        results['elapsed_time'] = elapsed_time
        results['timeout'] = timeout
        results['num_successes'] = len(success_indices)

        results['successes_per_second'] = results['num_successes']/results['elapsed_time'] 
        results['time_over_timeout'] = elapsed_time - timeout
        results['time_over_timeout_ratio'] = (elapsed_time - timeout)/(timeout + 1e-10)
        results['upload_bytes_mb'] =total_bytes_sent / 1000
        results['download_bytes_mb'] =total_bytes_recved / 1000
        results['upload_rate_mb'] =results['upload_bytes_mb']/elapsed_time 
        results['download_rate_mb'] =results['download_bytes_mb']/elapsed_time
        results['num_endpoints'] = num_endpoints
        results['success_rate'] = results['num_successes']/results['num_endpoints']

        # results['output_size'] = sys.getsizeof( results.pop['tensor'])
        results['batch_size'] = batch_size
        results['sequence_length'] = sequence_length
        results['num_tokens'] = batch_size*sequence_length

        for is_success in [True, False]:
            for m in ['min', 'max', 'mean', 'std']:
                if is_success:
                    results[f'success_latency_{m}'] = getattr(torch, m)(results['latency'][success_indices]).item()
                else:
                    results[f'latency_{m}'] = getattr(torch, m)(results['latency']).item()



        result_keys = ['tensor', 'latency', 'code', 'uid']

        # results['code'] = list(map(self.errorcode2name, results['code'].tolist()))


        graph_state_dict = self.graph.state_dict()
        graph_keys = ['trust', 'consensus','stake', 'incentive', 'dividends', 'emission']
        for k in graph_keys:
            results[k] =  graph_state_dict[k][results['uid']]
        
        if success_only:
            for k in result_keys + graph_keys:
                results[k] = results[k][success_indices]


        if return_type in ['metric', 'metrics']:
            results = {k:v for k,v  in results.items() if k not in graph_keys+result_keys }

        elif return_type in ['results', 'result']:
            results = {k:v for k,v  in results.items()\
                             if k not in (graph_keys+result_keys) }

        else:
            raise Exception(f'{return_type} not supported')
        return results
    def process_results(self, results, ):
        results_dict = {'tensor':[], 'code':[], 'latency':[], 'uid': []}

        num_responses = len(results[0])
        for i in range(num_responses):
            tensor = results[0][i][0]
            code = results[1][i][0]
            latency = results[2][i][0]
            endpoint = results[3][i]

            results_dict['tensor'].append(tensor)
            results_dict['code'].append(code)
            results_dict['latency'].append(latency)
            results_dict['uid'].append(endpoint)

        if len(results_dict['tensor'])>0:
            results_dict['tensor'] = torch.stack(results_dict['tensor'])
            results_dict['code'] = torch.tensor(results_dict['code'])
            results_dict['latency'] = torch.tensor(results_dict['latency'])
            results_dict['uid'] = torch.tensor(results_dict['uid'])
        else:
            results_dict['tensor'] = torch.tensor([])
            results_dict['code'] = torch.tensor([])
            results_dict['latency'] = torch.tensor([])
            results_dict['uid'] =  torch.tensor([])

        return results_dict


    # def run_experiment()



    def run_experiment(self,
            params = dict(
                sequence_length=[16,32,64],
                batch_size=[4,8,16,32],
                num_endpoints=[32,64,128, 256, 512],
                timeout=[2,4,6,8,10, 12],
                synapse=['TextLastHiddenState']
            ),
            experiment='experiment3',
            sequence_length=[]):

        # def flatten_hyperparams(hyperparams, flat_list =[]):
        #     for k,v_obj in hyperparams.items():
        #         tmp_params = deepcopy(hyperparams)
        #         if isinstance(v_obj, list):
        #             for v in v_obj:
        #                 tmp_params[k] = v
        #                 flat_list += flatten_hyperparams(hyperparams=tmp_params[k], flat_list=flat_list)
        #         else:
        #             continue

            
        sample_kwargs_list = []
        for sequence_length in params['sequence_length']:
            for num_endpoints in params['num_endpoints']:
                for timeout in params['timeout']:
                    for synapse in params['synapse']:
                        for batch_size in params['batch_size']:
                            sample_kwargs_list += [dict(
                                sequence_length = sequence_length,
                                batch_size = batch_size,
                                timeout= timeout,
                                synapse = synapse,
                                num_endpoints = num_endpoints,
                                success_only= False,
                                return_type='metric'
                            )]
        random.shuffle(sample_kwargs_list)
        for i,sample_kwargs in enumerate(tqdm(sample_kwargs_list)):
            self.set_receptor_pool(refresh=True)
            trial_metrics_result = self.sample(**sample_kwargs)
            self.put_json(f'{experiment}_{i}', trial_metrics_result)
  
    # def streamlit(self):
    #     for k,v_list in params.items():
    def streamlit(self):
        st.write('fam')
        st.write(self.run_experiment())






    def load_experiment(self, path='experiments'):
        df = []
        
        for p in self.glob_json(path+'*'):
            df.append(self.client.local.get_json(p))

        df =  pd.DataFrame(df)

        # df = pd.concat(df)
        # returnid2code = {k:f'{v}' for k,v in zip(bittensor.proto.ReturnCode.values(),bittensor.proto.ReturnCode.keys())}
        # df['code'] = df['code'].map(returnid2code)
        return df

    def streamlit_experiment(experiment= 'experiment'):
        df = module.load_experiment(path=experiment)
        from commune.streamlit import StreamlitPlotModule, row_column_bundles
    
        StreamlitPlotModule().run(df)
if __name__ == '__main__':
    # Sandbox.ray_restart()
    Module.new_event_loop()
    module = Sandbox.deploy(actor=False, load=True)
    module.streamlit()
