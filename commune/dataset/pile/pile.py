

import commune
from typing import List
import json
        
import threading
import queue
import os
import torch


class Pile(commune.Module):
    num_shards = 29
    default_shards = list(range(num_shards))
    
    def __init__(self,config=None):
        self.stop_threads = False
        self.device = 'cpu'

        config = self.set_config(config)
        self.url = self.config.url
        self.set_shards(config.shards)
        self.set_tokenizer(config.tokenizer)
        self.start_text_generator()
    
    
    def set_shards(self, shards):
        self.shards_urls = self.get_shard_urls(shards)
            
    @classmethod
    def resolve_shards(self, shards):
        if isinstance(shards, int):
            shards = list(range(shards))
        assert isinstance(shards, list)
        
        for s in shards:
            assert isinstance(s, int)
            
        return shards

    @classmethod
    def get_shard_urls(cls, shards: List[int] = 29, split='train'):
        shards = cls.resolve_shards(shards)
        shard_urls = []
        for s in shards:
            shard_urls.append(cls.get_shard_url(s, split=split))
        return shard_urls
        
    
    
    @classmethod
    def get_shard_url(cls, shard=0, split='train'):
        config = cls.config()
        assert isinstance(shard, int)
        filename =f'{shard}.jsonl.zst' if  bool(shard >= 10) else f'0{shard}.jsonl.zst'
        shard_url = f'{config.url}/{split}/{filename}'
        return shard_url
    
    @classmethod
    def ls_shards(cls):
        return [p for p in cls.glob('shards') if p.endswith('.jsonl')]
    
    @classmethod
    def get_shard_path(cls, shard:int, split:str='train', ext='jsonl'):
        filename = f'{shard}' if shard >= 10 else f'0{shard}'
        path= cls.resolve_path(f'shards/{filename}.{ext}')
        return path

    resolve_shard_path = get_shard_path
    @classmethod
    def shard_exists(cls, shard:int, split='train')-> bool:
        shard_path = cls.resolve_shard_path(shard,split)
        return bool(shard_path in cls.ls_shards())
    
    
    @classmethod
    def download_shard(cls, 
                       shard:int = 1,
                       split='train', 
                       refresh: bool = False,
                       *args, **kwargs):
        shard_url = cls.get_shard_url( shard=shard, split=split)
        shard_exists = cls.shard_exists(shard=shard, split=split)
        path = cls.resolve_shard_path(shard, split)
            
        if shard_exists and not refresh :
            cls.print(f'THE PILE: shard {shard} for split {split} exists', color='yellow')
            return None
        
        return cls.cmd(f'wget -P {path} {shard_url}', verbose=True, *args, **kwargs)
        
    @classmethod
    def download_fleet(cls, shards=3, split='train'):
        for s in range(shards):
            name = f'task.pile.download.s{s}.{split}'
            cls.deploy(fn='deploy_shard', name=name )
            
    

    def get_text(self, shard=1, split='train', path=None, start_pos=0):
        path = self.get_shard_path(shard=shard, split=split) if path is None else path
        with open(path, 'r') as f:
            # Move the file pointer to the starting position
            f.seek(start_pos)
            cnt = 0
            for line in f:
                # print(line)
                data = json.loads(line)

                # # print(data['text'])
                # self.print(data['text'])
                self.queue.put(data['text'])
                if self.stop_threads:
                    break
                

    def start_text_generator(self, num_threads=1, shard=1, split='train', path=None):
        self.queue = queue.Queue(1000)
        path = self.get_shard_path(shard=shard, split=split)
        file_size = os.stat(path).st_size
        chunk_size = file_size // num_threads
        start_pos = 0
        threads = []
        for i in range(num_threads):
            # Start the thread with the current start position
            t = threading.Thread(target=self.get_text, args=(shard, split, path, start_pos))
            t.start()
            threads.append(t)
            # Update the start position for the next thread
            start_pos += chunk_size
            # If this is the last thread, read until the end of the file
            if i == num_threads - 2:
                chunk_size = file_size - start_pos
                
        self.threads = threads
        # # Wait for all threads to finish
        # for t in threads:
        #     t.join()
            
    
    def stop_threads(self):
        self.stop_threads=True


    def __del__(self):
        self.shutdown()

    def shutdown(self, wait=True):
        self.stop_threads = True
        # if wait:
        #     for t in self.threads:
        #         try:
        #             t.join()
        #         except Exception:
        #             pass
    
    def sample_text(self):
        return self.queue.get()
    
    def sample(self, batch_size:int=32, sequence_length:int=256, idx_list:List[int] = None, tokenize:bool= True)->dict:
        
        sample_dict = {'text': [self.sample_text() for i in range(batch_size)]}
            
        if tokenize:
            sample_dict = self.tokenize(text=sample_dict['text'], max_length=sequence_length)

        return sample_dict
    
    forward = sample
    
    
    def tokenize(self, text: str = 'Whadup',
                 padding=True, 
                 truncation=True, 
                 max_length=256,

                 return_tensors='pt',
                 add_special_tokens=False,
                 device:str = None,
                 tokenizer: str = None, 
                 **kwargs) -> torch.Tensor:
        """ Returns tokenized text as torch tensor. """
        tokenizer = tokenizer if tokenizer else self.tokenizer
        if isinstance(tokenizer, str):
            raise NotImplementedError
        sample = tokenizer(text, 
                                             padding=padding, 
                                             truncation=truncation, 
                                             max_length=max_length, 
                                             return_tensors=return_tensors,
                                             add_special_tokens=add_special_tokens, 
                                             **kwargs)  # assume tokenizer.padding_side = 'left'

        device = device if device != None else self.device
        
        sample = dict(
            input_ids= sample['input_ids'].to(device),
            attention_mask= sample['attention_mask'].to(device)
        )
        
        return sample



    def set_tokenizer(self, tokenizer):
        from transformers import AutoTokenizer, AutoModel
        from commune.utils.tokenizer import prep_tokenizer
            
        assert isinstance(tokenizer, str)
        self.print(f'setting {tokenizer} tokenizer...')
        assert isinstance(tokenizer, str, )
        self.config['tokenizer'] = tokenizer
        
        try:
            # HACK TO INCLUDE LLAMA TOKENIZER
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast= True)
        except ValueError:

            print('resorting ot use_fast = False')
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
            
        self.tokenizer = tokenizer
        self.std_tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast= True)
        self.std_tokenizer = prep_tokenizer(self.std_tokenizer)
        self.tokenizer = prep_tokenizer(self.tokenizer, self.std_tokenizer)

        return self.tokenizer

    def to(self, device):
        self.device = device
        return self.device

    @classmethod
    def test(cls, *args, **kwargs):
        self = cls(*args,**kwargs)
        self.print(self.sample())
        self.shutdown()

        # self.shutdown()
