
import os, sys
import pandas as pd
from typing import Union, List
import plotly.express as px
from huggingface_hub import HfApi, hf_hub_download
import transformers
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset, load_dataset_builder
import commune
import streamlit as st
import torch
from safetensors.torch import save_file, load_file


shortcuts =  {
    # 0-1B models
    'gpt125m': 'EleutherAI/gpt-neo-125m',

    # 1-3B models
    'gpt2.7b': 'EleutherAI/gpt-neo-2.7B',
    'gpt3b': 'EleutherAI/gpt-neo-2.7B',
    'opt1.3b': 'facebook/opt-1.3b',
    'opt2.7b': 'facebook/opt-2.7b',
    # 'gpt3btuning' : ''

    # 0-7B models
    'gptjt': 'togethercomputer/GPT-JT-6B-v1',
    'gptjt_mod': 'togethercomputer/GPT-JT-Moderation-6B',
    'gptj': 'EleutherAI/gpt-j-6b',
    'gptj.pyg6b': 'PygmalionAI/pygmalion-6b',
    'gpt6b': 'cerebras/Cerebras-GPT-6.7B',
    'gptj.instruct': 'nlpcloud/instruct-gpt-j-fp16',
    'gptj.codegen': 'moyix/codegen-2B-mono-gptj',
    'gptj.hivemind': 'hivemind/gpt-j-6B-8bit',
    'gptj.adventure': 'KoboldAI/GPT-J-6B-Adventure',
    'gptj.pygppo': 'TehVenom/GPT-J-Pyg_PPO-6B', 
    'gptj.alpaca.gpt4': 'vicgalle/gpt-j-6B-alpaca-gpt4',
    'gptj.alpaca': 'bertin-project/bertin-gpt-j-6B-alpaca',
    'oa.galactia.6.7b': 'OpenAssistant/galactica-6.7b-finetuned',
    'opt6.7b': 'facebook/opt-6.7b',
    'llama': 'decapoda-research/llama-7b-hf',
    'vicuna.13b': 'lmsys/vicuna-13b-delta-v0',
    'vicuna.7b': 'lmsys/vicuna-7b-delta-v0',
    'llama-trl': 'trl-lib/llama-7b-se-rl-peft',
    'opt.nerybus': 'KoboldAI/OPT-6.7B-Nerybus-Mix',
    'pygmalion-6b': 'PygmalionAI/pygmalion-6b',
    # # > 7B models
    'oa.pythia.12b': 'OpenAssistant/oasst-sft-1-pythia-12b',
    'gptneox': 'EleutherAI/gpt-neox-20b',
    'gpt20b': 'EleutherAI/gpt-neox-20b',
    'opt13b': 'facebook/opt-13b',
    'gpt13b': 'cerebras/Cerebras-GPT-13B',
    
        }


class Huggingface(commune.Module):
    
    shortcuts = shortcuts
    def __init__(self, config:dict=None):
        self.set_config(config)
        self.hf_api = HfApi(self.config.get('hub'))
 
 
 
    @classmethod
    def get_tokenizer(cls, model:str=None, *args, **kwargs):
        model = cls.resolve_model(model)
        return AutoTokenizer.from_pretrained(model, *args, **kwargs)

    def get_model(self, model_name_or_path:str=None, *args, **kwargs):
        model = cls.resolve_model(model)
        return AutoModel.from_pretrained(model_name_or_path, *args, **kwargs)
 
 
    @classmethod
    def resolve_model(cls, model:str, *args, **kwargs):
        if model in cls.shortcuts:
            model = cls.shortcuts[model]
        return model
    
    @classmethod
    def model_saved(cls, model:str):
        model = cls.resolve_model(model)
        return bool(model in cls.saved_models())
    
    def list_datasets(self,return_type = 'dict', filter_fn=lambda x: x['downloads'] > 1000, refresh_cache =False, *args, **kwargs):
        datasets = {} if refresh_cache else self.get_json('datasets',default={})
        if len(datasets) == 0:
            datasets =  self.hf_api.list_datasets(*args,**kwargs)
            filter_fn = self.resolve_filter_fn(filter_fn=filter_fn)
            datasets = list(map(lambda x: x.__dict__, datasets))
            self.put_json('datasets',datasets)

        # st.write(datasets)
        df = pd.DataFrame(datasets)
        

        df['tags'] = df['tags'].apply(lambda tags: {tag.split(':')[0]:tag.split(':')[1] if len(tag.split(':')) == 2 else tag for tag in tags  }).tolist()
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
    
    cache_path = '~/.cache/huggingface'
    @classmethod
    def models(cls, limit=10, return_paths: bool = False , **kwargs):
        paths = [p for p in cls.model_paths()]
        
        if return_paths:
            paths = paths
        else:
            paths =  [p.split('models--')[-1].replace('--', '/') for p in paths]
            
        paths = [p for p in paths if '/' in p] 
        return paths
    @classmethod
    def snapshot_download(cls,repo_id, *args, **kwargs):
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id, *args, **kwargs)
    @classmethod
    def download(cls,model, *args, **kwargs):
        model = cls.resolve_model(model, *args, **kwargs)
        return cls.snapshot_download(model, *args, **kwargs)
    @classmethod
    def model_paths(cls, limit=10, **kwargs):
        dirpath = f'{cls.cache_path}/hub'
        
        return [p for p in commune.ls(dirpath) if os.path.basename(p).startswith('models')]
    
    
    
    
    @classmethod
    def saved_model2path(cls,  **kwargs):
        paths = [p for p in cls.model_paths()]
        model2path = {}
        for path in paths:
            model_name =os.path.basename(path).split('models--')[-1].replace('--', '/')
            model2path[model_name] = path
        print(model2path)
        return model2path
    
    
    @classmethod
    def saved_models(cls) -> List[str]:
        return list(cls.saved_model2path().keys())
    
    @classmethod
    def get_model_snapshots(cls, model):
        model = cls.resolve_model(model)
        root_path = cls.saved_model2path().get(model) + '/snapshots'
        snapshots = commune.ls(root_path)
        return [ snapshot  for snapshot in snapshots]
    
    @classmethod
    def get_model_snapshot(cls, model):
        snapshots = cls.get_model_snapshots(model) 
        return snapshots[0]
    
    get_model_path = get_model_snapshot
    
    @classmethod
    def get_model_assets(cls, model, search=None):
        model = cls.resolve_model(model)
        snapshots = cls.get_model_snapshots(model) 
        asset_paths = cls.ls(snapshots[0])
        if search != None:
            asset_paths = [a for a in asset_paths if search in a]
        
        return asset_paths
    
    @classmethod
    def get_model_config(cls, model_name):
        snapshots = cls.get_model_assets(model_name) 
        config_path = None
        for asset_path in snapshots:
            if asset_path.endswith('config.json'):
                config_path = asset_path
                break
            
        if config_path == None:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(model_name)
        assert config_path != None
        config = cls.load_json(config_path)
            
        return config
        
        
    @classmethod
    def load_torch(cls, path):
        ext = os.path.splitext(path)[-1]
        if 'safetensors' in ext:
            torch_data = load_file(path)
        else:
            torch_data = torch.load(path)
        assert isinstance(torch_data, dict) or isinstance(torch_data, torch.Tensor)
        return torch_data
    @classmethod
    def get_model_weights(cls,
                          model_name = None, 
                          load:bool = False,
                          mode = 'safetensors'):
        model_name = cls.resolve_model(model_name)
        asset_paths = cls.get_model_assets(model_name) 
        model_weight_paths = []
        model_weights = {}
        for asset_path in asset_paths:
            ext = os.path.splitext(asset_path)[-1]
            if mode not in ext:
                continue
            model_weight_paths.append(asset_path)
            if load:
                model_weights_chunk = cls.load_torch(asset_path)
                model_weights.update(model_weights_chunk)
                                      
        if load:
            return model_weights
        return model_weight_paths
    
    

    @classmethod
    def load_model_weights(cls, model='gpt20b',remove_prefix=True):
        
        model = cls.resolve_model(model)
        state_dict = {}
        weight_paths = cls.get_model_weights(model, load=False)
        weight_paths = sorted(weight_paths)

        for i,weight_path in enumerate(weight_paths):
            chunk_state_dict = cls.load_torch(weight_path)
            if remove_prefix:
                chunk_state_dict = {'.'.join(k.split('.')[1:]):v for k,v in chunk_state_dict.items()}
            state_dict.update(chunk_state_dict)
            
            
        return state_dict


        
    @classmethod
    def test(cls): 
        self = cls()
        cls.print(self.download('chavinlo/gpt4-x-alpaca'))

    @classmethod
    def class_init(cls):
        global Huggingface
        import huggingface_hub
        Huggingface = cls.merge(huggingface_hub)
        return Huggingface
    
    
    def get_model_config(self, model_name):
        return self.get_model_config(model_name)
    
    @classmethod
    def test(cls):
        cls.model_paths()

Huggingface.class_init()
if __name__ == '__main__':
    Huggingface.run()