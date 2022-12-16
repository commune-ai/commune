
from __future__ import print_function
##################
##### Import #####
##################
import torch
import concurrent.futures
import time
import psutil
import sys
import random
import argparse
from tqdm import tqdm
import asyncio
import pandas as pd
asyncio.set_event_loop(asyncio.new_event_loop())
import bittensor
import glob
import queue
import streamlit as st
import numpy as np
import aiohttp
import json
import os
from fsspec.asyn import AsyncFileSystem, sync, sync_wrapper
from commune import Module
##########################
##### Get args ###########
##########################
from typing import *
from munch import Munch

from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

import commune.sandbox.dataset.constant as constant

class DatasetTesting:


    def run_trial(self,  
                block_size=1000, 
                sequence_length=256,
                max_blocks_per_dataset=10e9,
                 batch_size=32, 
                 dataset_name = 'default', 
                 load_dataset=False, 
                 save_dataset=True, 
                 cache_size: int = 50, 
                 num_samples=1000,
                 cache_calls_per_update=10000,
                 ):
        dataset = bittensor.dataset( block_size=block_size, sequence_length=sequence_length, 
                                batch_size=batch_size, dataset_name = dataset_name,
                                max_blocks_per_dataset=max_blocks_per_dataset,
                                cache_calls_per_update=cache_calls_per_update,
                                 load_dataset=load_dataset, save_dataset=save_dataset,
                                  cache_size = cache_size, 
                                    )
        
        next(dataset)
        with Module.timer() as t:
            for i in range(num_samples):
                # st.write(Module.get_memory_info())
                # st.write('dataset size: ',total_size(dataset.__dict__))
                # st.write('Write')
                next(dataset)
                st.write('', i / t.elapsed_time.total_seconds())
    

    @classmethod
    def test_change_data_size(cls):
        data_sizes = [(10,1000), (15, 2000),(30, 3000), (25,4000)]
        dataset = bittensor.dataset(num_batches = constant.dataset.num_batches, dataset_name = constant.dataset.dataset_name, no_tokenizer=False)
        for data_size in data_sizes:
            dataset.set_data_size(*data_size)
            sample_dict = next(dataset)
            for k,v in sample_dict.items():
                v.shape[0] == data_size[0]
            
        dataset = bittensor.dataset(num_batches = constant.dataset.num_batches, dataset_name = constant.dataset.dataset_name, no_tokenizer=True)

        for data_size in data_sizes:
            raw_text_sample = next(dataset)
            len(raw_text_sample)  == data_size[1]
        
        dataset.close() 


    def run_experiment(self,
                        params=dict(
                            block_size= [1000, 5000, 10000, 20000],
                            sequence_length = [64, 128, 256, 512],
                            batch_size = [16,32, 64, 128],
                         ) ):
        pass




    @staticmethod
    def test_next_tokenized_sample():
        batch_size = 10
        sequence_length = 128
        block_size = 1000
        num_batches = 10


        dataset = bittensor.dataset (
            block_size = block_size,
            batch_size = batch_size,
            sequence_length = sequence_length,
            num_batches=num_batches,
            no_tokenizer=False
        )

        for i in range(num_batches):
            input = next(dataset)
            assert input['input_ids'].shape[0] == input['attention_mask'].shape[0] == batch_size
            assert input['input_ids'].shape[1] == input['attention_mask'].shape[1] == sequence_length
            dataset.close()

    @staticmethod
    def test_change_data_size():
        data_sizes = [dict(batch_size=10,sequence_length=64, block_size=1000, buffer_size=2000),
                            dict(batch_size=20,sequence_length=128, block_size=2000, buffer_size=3000),
                            dict(batch_size=30,sequence_length=256, block_size=4000, buffer_size=1000) ]
        dataset = bittensor.dataset(num_batches = constant.dataset.num_batches, dataset_name = constant.dataset.dataset_name, no_tokenizer=False)
        for data_size in data_sizes:
            dataset.set_data_size(**data_size)
            st.write(data_size)
            dataset.no_tokenizer = False
            sample = next(dataset)
            assert sample.shape[0] == data_size['batch_size']
            assert sample.shape[1] == data_size['sequence_length']
            assert dataset.block_size == data_size['block_size']
            assert dataset.buffer_size == data_size['buffer_size'] == len(dataset.sample_buffer)
            dataset.no_tokenizer = True
            sample = next(dataset)
            assert len(sample)== data_size['batch_size']
            assert len(sample[0].split()) == data_size['sequence_length']

        dataset.close() 



    @staticmethod
    def test_next_raw_sample():
        batch_size = 10
        sequence_length = 128
        block_size = 1000
        num_batches = 10
        dataset = bittensor.dataset (
            block_size = block_size,
            batch_size = batch_size,
            sequence_length = sequence_length,
            num_batches=num_batches,
            no_tokenizer = True
        )

        input = next(dataset)
        assert len(input) == batch_size
        for i in range(len(input)):
            assert len(input[i].split()) == sequence_length

        dataset.close()

    @staticmethod
    def test_speed(dataset_class, dataset_kwargs={}, steps=10, experiment='experiment_compare', refresh=False):
        
        path = f'/tmp/experiments/old/{experiment}.json'
        if commune.path_exists(path) and refresh == False:
            return pd.DataFrame(commune.get_json(path))
    
        dataset = dataset_class(**dataset_kwargs)
        with commune.timer() as t:
            cnt = 0
            previous_seconds =  0
            time_log_df = []
            for i in range(steps):
                raw_text_sample = next(dataset)
                seconds = t.elapsed_time.total_seconds() 
                row_dict  = dict(count=i, seconds=seconds, rate=i/seconds)
                st.write(row_dict)
                time_log_df.append(row_dict)

        experiment_df =  pd.DataFrame(time_log_df)
        experiment_dict = pd.DataFrame(time_log_df).to_dict()
        commune.put_json(path,experiment_dict)

        return experiment_df



    @staticmethod
    def speed_comparison():
        mode = 'old'    
        speed_test_df_dict = {}
        for mode in ['old', 'new']:
            if mode=='new':
                dataset_class = bittensor.dataset
                dataset_kwargs = dict(block_size=4000,no_tokenizer=True, 
                                        buffer_size=2000, sequence_length=256, batch_size=32, buffer_calls_per_update=20,
                                        dataset_name = ['ArXiv'], load_dataset=False, save_dataset=True)
                speed_test_df_dict[mode] = DatasetTesting.test_speed(dataset_class=dataset_class, dataset_kwargs=dataset_kwargs, refresh=False, steps=1000)
            elif mode=='old':
                dataset_class = bittensor.old_dataset
                dataset_kwargs = dict(dataset_name = ['ArXiv'])
                speed_test_df_dict[mode] = DatasetTesting.test_speed(dataset_class=dataset_class, dataset_kwargs=dataset_kwargs, experiment='old_dataset', refresh=False, steps=1000)
        
        import plotly.express as px


        fig_map = {}
        import plotly.graph_objects as go

        dfs  = []
        for mode in ['old', 'new']:
            df = speed_test_df_dict[mode]
            df['mode'] = mode
            dfs.append(df)

        df = pd.concat(dfs)

        fig = px.line(df, x='seconds', y='count', color='mode')
        fig.update_layout(title=' Race to 1000 batches with Old and New Dataset')

        st.write(fig)
        

    @staticmethod
    def test_conflict_rate(n_samples=1000):
        data = bittensor.dataset(dataset_name=['ArXiv'] , save_dataset=False, load_dataset=False,  buffer_calls_per_update=100, buffer_size=4000)
        item = next(data)

        st.write(len(data.sample_buffer[0]))
        st.write(len(set([fm['Size']for fm in data.all_text_file_metas])))
        

        all_text = []
        st.write(f"{data.buffer_calls_per_update}")
        for _ in tqdm(range(n_samples)):
            st.write(_)
            texts = [data.tokenizer.decode(text) for text in item]
            all_text += texts
            item = next(data)


        st.write(f"{len(set(all_text))} / {len(all_text)} unique")

if __name__ == '__main__':

    st.write('## NEW DATASET')
    # DatasetTesting.test__data_size()
    # DatasetTesting.test_conflict_rate()
    # DatasetTesting.test_change_data_size()
    # st.write(DatasetTesting().run_trial())
    data = bittensor.dataset(dataset_name=['ArXiv'] , save_dataset=True, load_dataset=False,  buffer_calls_per_update=100, buffer_size=4000)

    st.write(next(data))


