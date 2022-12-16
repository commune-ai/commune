
import os, sys
import datasets 
import datetime
import pandas as pd
import transformers
from copy import deepcopy
from typing import Union, List
from copy import deepcopy
from commune import Module
import torch
import ray
from commune.utils import dict_put
from datasets.utils.py_utils import asdict, unique_values
import datetime
import plotly.express as px
from huggingface_hub import HfApi
from ocean_lib.models.data_nft import DataNFT
import fsspec
import os
from ipfsspec.asyn import AsyncIPFSFileSystem
from fsspec import register_implementation
import asyncio
import io
from commune.utils import *
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset, load_dataset_builder



class HubModule(Module):
    
    def __init__(self):
        Module.__init__(self)
        self.hf_api = HfApi(self.config.get('hub'))
 
    def list_datasets(self,return_type = 'dict', filter_fn=None, refresh_cache =False, *args, **kwargs):
        datasets = {} if refresh_cache else self.get_json('datasets',default={})
        if len(datasets) == 0:
            datasets =  self.hf_api.list_datasets(*args,**kwargs)
            filter_fn = self.resolve_filter_fn(filter_fn=filter_fn)
            datasets = list(map(lambda x: x.__dict__, datasets))
            self.put_json('datasets',datasets)

        df = pd.DataFrame(datasets)
        df['num_tags'] = df['tags'].apply(len)
        df['tags'] = df['tags'].apply(lambda tags: {tag.split(':')[0]:tag.split(':')[1] for tag in tags  }).tolist()
        for tag_field in ['task_categories']:
            df[tag_field] = df['tags'].apply(lambda tag:tag.get(tag_field) )
        df['size_categories'] = df['tags'].apply(lambda t: t.get('size_categories'))
        df = df.sort_values('downloads', ascending=False)
        if filter_fn != None and callable(filter_fn):
            df = self.filter_df(df=df, fn=filter_fn)


        return df
    

    @staticmethod
    def resolve_filter_fn(filter_fn):
        if filter_fn != None:
            if callable(filter_fn):
                fn = filter_fn

            if isinstance(filter_fn, str):
                filter_fn = eval(f'lambda r : {filter_fn}')
        
            assert(callable(filter_fn))
        return filter_fn

    def list_models(self,return_type = 'pandas',filter_fn=None, *args, **kwargs):
        
        
        models = self.get_json('models',default={})
        if len(models) == 0:
            models = self.hf_api.list_models(*args,**kwargs)
            filter_fn = self.resolve_filter_fn(filter_fn=filter_fn)
            models = list(map(lambda x: x.__dict__, models))
            self.put_json('models', models)
        models = pd.DataFrame(models)
        if filter_fn != None and callable(filter_fn):
            models = self.filter_df(df=models, fn=filter_fn)

        return models

    @property
    def models(self):
        df = self.list_models(return_type='pandas')
        return df
    @property
    def datasets(self):
        df = self.list_datasets(return_type='pandas')
        return df

    @property
    def task_categories(self):
        return list(self.datasets['task_categories'].unique())
    @property
    def pipeline_tags(self): 
        df = self.list_models(return_type='pandas')
        return df['pipeline_tag'].unique()
    @property
    def pipeline_tags_count(self):
        count_dict = dict(self.models['pipeline_tag'].value_counts())
        return {k:int(v) for k,v in count_dict.items()}

    def streamlit_main(self):
        pipeline_tags_count = module.pipeline_tags_count
        fig = px.pie(names=list(pipeline_tags_count.keys()),  values=list(pipeline_tags_count.values()))
        fig.update_traces(textposition='inside')
        fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
        st.write(fig)
    def streamlit_sidebar(self):
        pass

    def streamlit(self):
        self.streamlit_main()
        self.streamlit_sidebar()


    def streamlit_datasets(self, limit=True):
        with st.expander('Datasets', True):
            st.write(self.datasets.iloc[0:limit])

    def streamlit_models(self, limit=100):
        with st.expander('Models', True):
            st.write(self.models.iloc[0:limit])

    def streamlit_dfs(self, limit=100):
        self.streamlit_datasets(limit=limit)
        self.streamlit_models(limit=limit)

    def dataset_tags(self, limit=10, **kwargs):
        df = self.list_datasets(limit=limit,return_type='pandas', **kwargs)
        tag_dict_list = df['tags'].apply(lambda tags: {tag.split(':')[0]:tag.split(':')[1] for tag in tags  }).tolist()
        tags_df =  pd.DataFrame(tag_dict_list)
        df = df.drop(columns=['tags'])
        return pd.concat([df, tags_df], axis=1)

    @staticmethod
    def filter_df(df, fn):
        indices =  df.apply(fn, axis=1)
        return df[indices]
if __name__ == '__main__':
    import streamlit as st
    import numpy as np

    module = HubModule()
    # st.write(module.datasets)

    st.write(module.datasets)
    st.write(module.models)

    # st.write(px.histogram(x=module.datasets.query('num_tags>=0 & num_tags<10')['num_tags']))
    # module.streamlit_main()
    