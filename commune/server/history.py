
import commune as c
from typing import *
import pandas as pd

class History(c.Module):
    def __init__(self, history_path='history', **kwargs):
        self.set_history_path(history_path)
    # HISTORY 
    def add_history(self, item:dict):    
        path = self.history_path + '/' + item['address'] + '/'+  str(item['timestamp']) 
        self.put(path, item)

    def set_history_path(self, history_path):
        assert history_path is not None, f"History path is not set"
        self.history_path = history_path
        return {'history_path': self.history_path}

    def rm_history(self, server):
        dirpath  = f'{self.history_path}/{server}'
        return c.rm(dirpath)
    
    @classmethod
    def history_paths(cls, server=None, history_path='history', n=100, key=None):
        if server == None:
            dirpath  = f'{history_path}'
            paths =  cls.glob(dirpath)
        else:
            
            dirpath  = f'{history_path}/{server}'
            paths =  cls.ls(dirpath)
        paths = sorted(paths, reverse=True)[:n]
        return paths

    def history(self, 
                key=None, 
                history_path='history',
                features=[ 'module', 'fn', 'seconds_ago', 'latency', 'address'], 
                to_list=False,
                **kwargs
                ):
        key = c.get_key(key)
        history_path = self.history_paths(key=key, history_path=history_path)
        df =  c.df([self.get(path) for path in history_path])
        now = c.timestamp()
        df['seconds_ago'] = df['timestamp'].apply(lambda x: now - x)
        df = df[features]
        if to_list:
            return df.to_dict('records')

        return df
    
    