import os
import sys
sys.path[0] = os.environ['PWD']
import numpy as np
import math
import requests
import json
import datetime
import pandas as pd
import asyncio
from munch import Munch
import argparse
asyncio.set_event_loop(asyncio.new_event_loop())

from typing import *
import bittensor
import streamlit as st
import plotly.express as px
from commune.config import ConfigLoader
from commune.utils import  chunk, dict_put, round_sig, deep2flat
import ray
import random
import torch
from copy import deepcopy
# function to use requests.post to make an API call to the submetagraph url
from commune import Module
from tqdm import tqdm
from plotly.subplots import make_subplots
from commune.ray.utils import kill_actor, create_actor
from ray.util.queue import Queue
import itertools
# from commune .process.extract.crypto.utils import run_query
# from commune.plot.dag import DagModule
from commune.streamlit import StreamlitPlotModule, row_column_bundles


class BittensorModule(Module):

    default_network = 'nakamoto'

    def __init__(self, 
                config: Union[str,dict, 'commune.Config']=None, 
                coldkey:str=None, 
                hotkey:str=None, 
                load:bool=True, 
                subtensor: 'bittensor.subtensor'=None,
                wallet: 'bittensor.wallet'=None,
                network:str = None,
                block:int=None,
                **kwargs):

        Module.__init__(self, config=config, **kwargs) 
        self.config.bittensor = self.get_bittensor_config()

        self.network = network = network if network else self.network

        self.set_wallet(wallet=wallet,hotkey=hotkey, coldkey=coldkey)  
        self.set_network(network=network, subtensor=subtensor, block=block)

        self.plot = StreamlitPlotModule()


    @staticmethod
    def str2synapse(synapse:str, *args, **kwargs):
        return getattr(bittensor.synapse, synapse)(*args, **kwargs)
    @property
    def available_synapses(self):
        return [f for f in dir(bittensor.synapse) if f.startswith('Text')]

    def save(self, path=None ,**kwargs):
        self.put_config(path=path, **kwargs)


    def set_wallet(self, coldkey:str=None, hotkey:str=None,**kwargs):

        coldkey = self.config.wallet.coldkey = coldkey if coldkey  else  self.config.wallet.coldkey
        hotkey =self.config.wallet.hotkey = hotkey if hotkey else self.config.wallet.hotkey 

        self.wallet = bittensor.wallet(name=coldkey, hotkey=hotkey)

        # for f in ['add_stake', 'remove_stake', 'get_balance']:
        #     fn = getattr(self.wallet, f)
        #     if callable(fn):
        #         setattr(self, f, fn)

        return self.wallet

    def get_endpoints(self, endpoint_ids=None , num_endpoints=None, random_sample=True):
        endpoints =self.metagraph.endpoint_objs
        selected_endpoints = []
        if isinstance(endpoint_ids, list ):
            for i in endpoint_ids:
                assert isinstance(i, int), i
                assert i > 0 and i < len(endpoints), endpoint_ids

                selected_endpoints.append(endpoints[i])

            return selected_endpoints
        
        if num_endpoints == None:
            num_endpoints =self.num_endpoints

        if random_sample == True:
            endpoint_index_list = list(np.random.randint(0, self.n, (num_endpoints)))
            selected_endpoints = [endpoints[idx] for idx in endpoint_index_list]
        else:
            selected_endpoints = endpoints[:num_endpoints]
        return selected_endpoints


    @property
    def block(self):
        if not hasattr(self, '_block'):
            self._block = self.current_block
        
        return self._block

    @block.setter
    def block(self, block):
        self._block = block
        self.config['block'] = block

    @property
    def neuron(self):
        return self.subtensor.neuron_for_pubkey(self.wallet.hotkey.ss58_address)
    
    @property
    def uid(self):
        return self.metagraph.hotkeys.index( self.wallet.hotkey.ss58_address )

    @property
    def coldkey_address(self):
        return self.wallet.coldkeypub.ss58_address

    @property
    def hotkey_address(self):
        return self.wallet.hotkey.ss58_address
    
    @property
    def endpoints(self):
        return self.metagraph.endpoint_objs

    @property
    def my_endpoints(self, mode = 'hotkey'):
        endpoints = self.metagraph.endpoint_objs
        
        if mode == 'hotkey':
            endpoints = [e for e in endpoints if (e.hotkey == self.hotkey_address and e.ip != "0.0.0.0") ]
        elif mode == 'coldkey':
            endpoints = [e for e in endpoints if (e.coldkey == self.coldkey_address and e.ip != "0.0.0.0") ]
        else:
            raise NotImplementedError

        return endpoints


    @property
    def subtensor(self):
        if not hasattr(self,'_subtensor'):
            self.set_subtensor(network=self.network)
        return self._subtensor




    @subtensor.setter
    def subtensor(self, subtensor):
        self._subtensor = subtensor



    def create_hotkey(self, hotkey, coldkey='default', register=False, **kwargs):
        hotkeys_wallets = self.wallets[coldkey]
        if hotkey in hotkeys_wallets:
            wallet =  hotkeys_wallets[hotkey]
        else:
            wallet = bittensor.wallet(hotkey=hotkey, name=coldkey)
            wallet.create_new_hotkey()

        if register:
            self.register(wallet=wallet)
        return wallet

    def is_registered(self, wallet):
        wallet = self.resolve_wallet(wallet=wallet)
        
        return wallet.is_registered(subtensor=self.subtensor)
        
    
    def list_hotkeys(self, coldkey='default'):
        return list(self.wallets[coldkey].keys())

    def list_coldkeys(self, coldkey='default'):
        return list(self.wallets.keys())



    @staticmethod
    def _get_hotkey_wallets_for_wallet( wallet ) -> List['bittensor.wallet']:
        hotkey_wallets = []
        hotkeys_path = wallet.path + '/' + wallet.name + '/hotkeys'
        try:
            hotkey_files = next(os.walk(os.path.expanduser(hotkeys_path)))[2]
        except StopIteration:
            hotkey_files = []
        for hotkey_file_name in hotkey_files:
            try:
                hotkey_for_name = bittensor.wallet( path = wallet.path, name = wallet.name, hotkey = hotkey_file_name )
                if hotkey_for_name.hotkey_file.exists_on_device() and not hotkey_for_name.hotkey_file.is_encrypted():
                    hotkey_wallets.append( hotkey_for_name )
            except Exception:
                pass
        return hotkey_wallets

    @staticmethod
    def _get_coldkey_wallets_for_path( path: str ) -> List['bittensor.wallet']:
        try:
            wallet_names = next(os.walk(os.path.expanduser(path)))[1]
            st.write(path, wallet_names)
            return [ bittensor.wallet( path= path, name=name ) for name in wallet_names ]
        except StopIteration:
            # No wallet files found.
            wallets = []
        return wallets

    @staticmethod
    def _get_all_wallets_for_path( path:str ) -> List['bittensor.wallet']:
        all_wallets = []
        cold_wallets = BittensorModule._get_coldkey_wallets_for_path(path)
        for cold_wallet in cold_wallets:
            if cold_wallet.coldkeypub_file.exists_on_device() and not cold_wallet.coldkeypub_file.is_encrypted():
                all_wallets.extend( BittensorModule._get_hotkey_wallets_for_wallet(cold_wallet) )
        return all_wallets


    def list_wallets(self, return_type='all', registered=False):
        wallet_path = self.wallet.config.wallet.path
        if return_type in ['coldkey', 'cold']:
            wallets = self._get_coldkey_wallets_for_path(path=wallet_path)
        elif return_type in ['hot', 'hotkey', 'all']:
            st.write(wallet_path)
            wallets  = self._get_all_wallets_for_path(path=wallet_path)
        else:
            raise NotImplementedError

        if registered:
            wallets = [w for w in wallets if self.is_registered(wallet=w)]
        else:
            wallets = [w for w in wallets if not self.is_registered(wallet=w)]
        

        return wallets
    @property
    def wallets(self):
        wallet_dict ={}

        for w in self.list_wallets():
            key = f'{w.name}.{w.hotkey_str}'
            dict_put(wallet_dict, keys=key, value=w)
        
        return wallet_dict


    @property
    def registered_wallets(self):
        wallet_dict ={}

        for w in self.list_wallets(registered=True):
            key = f'{w.name}.{w.hotkey_str}'
            dict_put(wallet_dict, keys=key, value=w)
        
        return wallet_dict

    @property
    def unregistered_wallets(self):
        wallet_dict ={}

        for w in self.list_wallets(registered=False):
            key = f'{w.name}.{w.hotkey_str}'
            dict_put(wallet_dict, keys=key, value=w)
        
        return wallet_dict


    def resolve_wallet(self, wallet):
        if wallet == None:
            wallet = self.wallet
        return wallet
    
    def wallet_exists(self, wallet):
        assert isinstance(wallet, str)

    def register(self, wallet=None, **kwargs):
        wallet = self.resolve_wallet(wallet=wallet)
        default_kwargs = dict(cuda=True)
        kwargs =  {**default_kwargs, **kwargs}
        return wallet.register(subtensor=self.subtensor, **kwargs)


    @property
    def network(self):
        if not hasattr(self, '_network'):
            self._network = self.config.get('network', self.default_network)
        return self._network

    @network.setter
    def network(self, network):
        assert network in self.networks, f'{network} is not in {self.networks}'
        self._network = network
        self.config['network'] = network
        return self._network

    @property
    def current_block(self):
        return self.subtensor.get_current_block()

    @property
    def n(self):
        return self.subtensor.n
    @property
    def max_n(self):
        return self.subtensor.max_n



    @property
    def metagraph_path(self):
        return f'backend/{self.network}B{self.block}.pth'


    def set_metagraph_state(self, sample_n=None, sample_mode='rank', **kwargs):
        metagraph_state = self.metagraph.state_dict()
        if sample_n != None:
            metagraph_state =  self.sample_metagraph_state(metagraph_state=metagraph_state, sample_n=sample_n, sample_mode=sample_mode, **kwargs)
        self.metagraph_state = metagraph_state


    sample_n = 400
    sample_mode = 'rank'
    sample_metric = 'ranks'
    sample_descending = True

    def sample_metagraph_state(self, metagraph_state , sample_n=None,  sample_mode='rank', **kwargs ):
        '''
        Args:
            sample_mode: 
                the method of sampling the neurons data. There are two 
        '''

        sample_n = sample_n if sample_n != None else self.sample_n
        sample_mode = sample_mode if sample_mode != None else self.sample_mode
        metric=kwargs.get('metric', self.sample_metric)
        descending = kwargs.get('descending', self.sample_descending)

        if sample_mode == 'rank':
            sampled_uid_indices = self.argsort_uids(metric=metric, descending=descending)[:sample_n]
        elif sample_mode == 'random':
            sampled_uid_indices = torch.randperm(self.n)[:sample_n]
        else:
            raise NotImplementedError


        self.sampled_uid_indices = sampled_uid_indices

        sampled_metagraph_state = {}
        for k,v in metagraph_state.items():
            if len(v.shape)==0:
                continue
            elif (len(v.shape) == 1 and v.shape[0] == self.n) or k in ['endpoints'] :
                sampled_metagraph_state[k] = v[sampled_uid_indices]
            elif len(v.shape) == 2 and v.shape[0] == self.n and v.shape[1] == self.n:
                sampled_metagraph_state[k] = v[sampled_uid_indices]
                sampled_metagraph_state[k] = sampled_metagraph_state[k][:, sampled_uid_indices]
            else:
                sampled_metagraph_state[k] = v
            
        return sampled_metagraph_state

    def argsort_uids(self, metric='rank', descending=True):
        prohibited_params = ['endpoints', 'uids', 'version']

        if metric in prohibited_params:
            return None

        metric_tensor = getattr(self.metagraph, metric, None)

        if metric_tensor == None :
            return None
        else:
            metric_shape  = metric_tensor.shape
            if len(metric_shape) == 2 and metric_shape[0] == self.n:
                metric_tensor = torch.einsum('ij->i', metric_tensor)
            if metric_shape[0] == self.n:
                return torch.argsort(metric_tensor, descending=descending, dim=0).tolist()    
    @property
    def blocks_behind_threshold(self):
        return self.config.get('blocks_behind_sync_threshold', 1)

    @property
    def sync_metagraph_bool(self):
        return self.blocks_behind

    @property
    def block(self):
        return self.metagraph.block.item()

    @property
    def blocks_behind(self):
        return self.current_block - self.block



    def set_network(self,  network:Optional[str]=None, block:Optional[int]=None, subtensor:Optional[str]=None,  **kwargs):
        '''
        The subtensor.network should likely be one of the following choices:
            -- local - (your locally running node)
            -- nobunaga - (staging)
            -- nakamoto - (main)
        '''
        # Load the subtensor graph
        self.subtensor = subtensor if subtensor else bittensor.subtensor(network=network, **kwargs)
        
        
        # Load the metagraph
        self.metagraph = bittensor.metagraph(network=self.network, subtensor=self.subtensor)
        self.metagraph.load()

        sync_metagraph_bool = self.sync_metagraph_bool or force_sync
        if sync_metagraph_bool:
            self.metagraph.sync(block=block)

        self.metagraph.save()
        self.set_metagraph_state()
    
    
    def adjacency(mode='W', ):
        return torch.nonzero(self.metagraph.weights)

    def describe_metagraph_state(self, shape=True):
        return {k:dict(shape=v.shape, type=v.dtype ) for k,v in  self.metagraph_state.items()}


    @property
    def metagraph_state_params(self):
        return self.metagraph_state.keys()

    def agg_param(self, param='rank', agg='sum', decimals=2):
        param_tensor = getattr(self.metagraph, param)
        return round(getattr(torch,agg)(param_tensor).item(), decimals)


    @property
    def networks(self):
        return bittensor.__networks__

    def st_select_network(self):

        with st.sidebar.form('Sync Network'):
            network2idx = {n:n_idx for n_idx, n in  enumerate(self.networks)}
            default_idx = network2idx['nakamoto']
            network = st.selectbox('Select Network', self.networks,default_idx)
            block = st.number_input('Select Block', 0,self.current_block, self.current_block-1)
            submitted = st.form_submit_button("Sync")

            if submitted:
                self.load(network=network, block=block)
                
        # ''')

    def st_metrics(self):

        cols = st.columns(3)
        # self.block = st.sidebar.slider('Block', 0, )
        cols[0].metric("Synced Block", f'{self.block}', f'{-self.blocks_behind} Blocks Behind')
        cols[1].metric("Network", self.network)
        cols[2].metric("Active Neurons ", f'{self.n}/{self.max_n}')


        metrics = [ 'trust', 'stake', 'consensus']
        fn_list = []
        fn_args_list = []
        from copy import deepcopy
        for metric in metrics:
            metric_show = 'Total '+ metric[0].upper() + metric[1:].lower()
            # st_fn = 
            metric_value = self.agg_param(metric)
            fn_args_list.append([metric_show, metric_value])
            fn_list.append(lambda name, value: st.metric(name, value ))

        row_column_bundles(fn_list= fn_list, fn_args_list=fn_args_list, cols_per_row=3)


    @property
    def rank(self):
        return self.neuron.rank

    @property
    def stake(self):
        return self.neuron.stake

    @property
    def consensus(self):
        return self.neuron.consensus

    @property
    def incentive(self):
        return self.neuron.incentive

    @property
    def emmision(self):
        return self.neuron.emmision

    @property
    def trust(self):
        return self.neuron.trust
    
    @property
    def uid_data(self):
        neuron = self.neuron
        return {'stake': neuron.stake,
                'rank': neuron.rank,
                'trust': neuron.trust,
                'consensus': neuron.consensus,
                'incentive': neuron.incentive,
                'emission': neuron.emission}

    def view_metagraph(self):

        n= 100
        edge_density = 0.1
        nodes=[dict(id=f"uid-{i}", 
                                label=f"uid-{i}", 
                                color='red',
                                size=100) for i in range(n)]

        edges = torch.nonzero(torch.rand(n,n)>edge_density).tolist()


        edges = [dict(source=f'uid-{i}', target=f'uid-{j}') for i,j in edges]



        self.plot.dag.build(nodes=nodes, edges=edges)
    
    def st_describe_metagraph_state(self):
        with st.sidebar.expander('metagraph Params'):
            st.write(self.describe_metagraph_state())

    def plot_sandbox(self):
        self.plot.run(data=self.metagraph.to_dataframe())
    
    def st_sidebar(self):
        bt_url = 'https://yt3.ggpht.com/9fs6F292la5PLdf-ATItg--4bhjzGfu5FlIV1ujfmqlS0pqKzGleXzMjjPorZwgUPfglMz3ygg=s900-c-k-c0x00ffffff-no-rj'
        bt_url = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAASwAAACoCAMAAABt9SM9AAAAVFBMVEUAAAD///8hISGxsbFgYGBcXFx1dXXn5+dUVFQrKysGBgYZGRmKiore3t6YmJinp6f39/fAwMDIyMhnZ2cQEBBsbGzv7+9BQUHX19fz8/N/f3/h4eHkVQerAAAA6klEQVR4nO3Yu27CQBRFURNjIAbCIwGc5P//M9gUCGi4Te4YrdV4yqNdjKWpKgAAAAAAAAAAAAAAAAAAAAAAAAB4IU32gIK1bzeWq3X2omLN5pN7H9mbSrV5SDXpsjeVqnlIta2zNxWr3jXTs8W+77Q6nxfZiwr22V6+dR/LX/A5Q6xp9oqRECtArACxAsQKECtArACxAsQKECtArIBrrK82e0vxhliH/nR8z95SvEMfa32qlrtj9pQR6IaHv/m3i+sZ3eWh9Cd7xzjM9r9d7cICAAAAAAAAAAAAAAAAAAAAAAAAAAAA4F/8AYjjA5tLvfqfAAAAAElFTkSuQmCC'
        st.sidebar.image(bt_url, width=300)
        # st.markdown('''
        # # BitDashboard
        # ---
        # ''')
        self.st_select_network()
        self.st_sample_params()
        self.st_describe_metagraph_state()

    def st_main(self):

        self.st_metrics()
        
        df = self.metagraph_df()
        with st.expander('metagraph Dataframe'):
            st.write(df)
        plot_df = df.drop(['uid', 'active'], axis=1)
        self.st_distributions(df = plot_df)
        self.st_scatter(df=plot_df)
        self.st_relationmap()

        
        with st.expander('Custom Plots'):
            self.plot.run(data=plot_df)

    def st_relationmap(self):
        with st.expander('Relation Map'):
            metric = st.selectbox('Select a Matric', ['weights', 'bonds'], 0)
            z = self.metagraph_state[metric]
            # z[torch.nonzero(z==1)] = 0
            cols =st.columns([1,5,1])

            fig = self.plot.imshow(z, text_auto=True, title=f'Relation Map of {metric.upper()}')
            fig.update_layout(autosize=True, width=800, height=800)
            cols[1].write(fig)
    def st_distributions(self, df):
        plot_columns = [c for c in df.columns if c not in ['uid', 'active']]


        with st.expander('Distibutions'):
            fn_list = [lambda col: st.write(self.plot.histogram(df, x=col, title=f'Distribution of {col.upper()}', color_discrete_sequence=random.sample(px.colors.qualitative.Plotly,1)))]*len(plot_columns)
            fn_args_list = [[col,] for col in plot_columns]
            row_column_bundles(fn_list=fn_list, fn_args_list=fn_args_list)

    response_id2code_map = {k:f'{v}' for k,v in zip(bittensor.proto.ReturnCode.values(),bittensor.proto.ReturnCode.keys())}


    def st_scatter(self, df):
        plot_columns = ['stake', 'rank', 'trust', 'consensus', 'incentive', 'dividends', 'emission']
        
        with st.expander('Scatter'):
            fn_list = []
            fn_args_list = []
            for col_x in plot_columns:
                for col_y in ['rank']:
                    if col_x != col_y:

                        fn_list += [lambda col_x, col_y: st.write(self.plot.scatter(df, x=col_x, y=col_y, title=f'{col_x.upper()} vs {col_y.upper()}', color_discrete_sequence=random.sample(px.colors.qualitative.Plotly,1)))]
                        fn_args_list += [[col_x,col_y]]
        
            row_column_bundles(fn_list=fn_list, fn_args_list=fn_args_list)


    def st_run(self):
        
        self.st_sidebar()
        self.st_main()

    
    def resolve_network(self, network:str):
        assert network in self.networks, f"{network} must be in {self.networks}"
        if network == 'main':
            network = 'nakamoto'

        # turn nakamoto to local if user wants to run local node
        if self.config.get('local_node') == True and network == 'nakamoto':
            network = 'local'



    def metagraph_df(self):
        df_dict= {
                'uid': self.metagraph_state['uids'],
                'active': self.metagraph_state['active'],             
                'stake': self.metagraph_state['stake'],             
                'rank': self.metagraph_state['ranks'],            
                'trust': self.metagraph_state['trust'],             
                'consensus': self.metagraph_state['consensus'],             
                'incentive': self.metagraph_state['incentive'],             
                'dividends': self.metagraph_state['dividends'],          
                'emission': self.metagraph_state['emission']
            }


        return pd.DataFrame(df_dict)

    def st_sample_params(self):
        with st.sidebar.form("sample_n_form"):
            self.sample_n = st.slider('Sample N', 1, self.n, self.sample_n )
            self.sample_mode = st.selectbox('Sample Mode', ['rank', 'random'], 0 )
            self.sample_metric = st.selectbox('Sample Metric', ['ranks', 'trust'], 0 )
            self.sample_descending = 'descending'== st.selectbox('Sample Metric', ['descending', 'ascending'], 0 )
            submitted = st.form_submit_button("Sample")

            if submitted:
                self.set_metagraph_state()


    @property
    def hotkey_address(self):
        return self.wallet.hotkey.ss58_address
    @property
    def endpoints(self):
        return self.metagraph.endpoint_objs
    @property
    def hotkey_endpoints(self):
        return self.my_endpoints(mode='hotkey')
    @property
    def coldkey_endpoints(self):
        return self.my_endpoints(mode='coldkey')

    ls_synapses = all_synapses = available_synapses
    
    @property
    def synapse_map(self):
        return {f:getattr(bittensor.synapse,f) for f in self.available_synapses}

    def get_synapse(self, synapse, *args, **kwargs):
        return self.synapse_map[synapse](*args, **kwargs)

    resolve_synapse = get_synapse

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

    


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    # st.write(module.blocks_behind)
    # st.write(module.register())

    parser = argparse.ArgumentParser()
    # ML model arguements
    parser.add_argument('--index', type=int, help='device i', default=0)
    args = parser.parse_args()

    actor = {'gpu': 0.2, 'cpu':1, 'name': 'bittensor_module-0', 'refresh': False, 'wrap': True}
    actor = False
    module = BittensorModule(hotkey=f'hotkey-{args.index}')
    module.wallet.create_new_hotkey(use_password=False, overwrite=False)
    st.write(module.register(dev_id=list(range(8))))


    

    # st.write(module.list_wallets())
    # st.write(module.register())
    # st.write(module.wallet.get_balance())
    # st.write(module.wallet.regenerate_coldkey(menmonic))