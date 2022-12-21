import datetime
from contextlib import contextmanager
def isoformat2datetime(isoformat:str):
    dt, _, us = isoformat.partition(".")
    dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
    us = int(us.rstrip("Z"), 10)
    dt = dt + datetime.timedelta(microseconds=us)
    assert isinstance(dt, datetime.datetime)
    return dt

def isoformat2timestamp(isoformat:str, return_type='int'):
    supported_types = ['int', 'float']
    assert return_type in supported_types, f'return type should in {supported_types} but you put {return_type}'
    dt = isoformat2datetime(isoformat)
    timestamp = eval(return_type)(dt.timestamp())
    assert isinstance(timestamp, int)
    return timestamp


def timedeltatimestamp( **kwargs):
    assert len(kwargs) == 1
    supported_modes = ['hours', 'seconds', 'minutes', 'days']
    mode = list(kwargs.keys())[0]
    assert mode in supported_modes, f'return type should in {supported_modes} but you put {mode}'
    
    current_timestamp = datetime.datetime.utcnow()
    timetamp_delta  =  current_timestamp.timestamp() -  ( current_timestamp- datetime.timedelta(**kwargs)).timestamp()
    return timetamp_delta



import time


class Timer:
    
    def __init__(self, text='time elapsed: {t}', return_type='seconds', streamlit=False, verbose=True ):   
        
        self.__dict__.update(locals())
        

    @property
    def start(self):
        self.local_start_time = self.seconds
        return self.local_start_time
    @property
    def stop(self):
        return self.seconds - self.local_start_time


    def __enter__(self):
        self.start_time = datetime.datetime.utcnow()
        return self

    @property
    def interval(self):
        self.end_time =  datetime.datetime.utcnow()
        interval = (self.end_time - self.start_time)

        return_type = self.return_type
        if return_type in ['microseconds', 'ms', 'micro', 'microsecond']:
            div_factor = 1
        elif return_type in ['seconds', 's' , 'second', 'sec']:
            div_factor = 1000
        
        elif return_type in ['minutes', 'm', 'min' , 'minutes']: 
            div_factor = 1000*60
        
        else:
            raise NotImplementedError
        

        return interval
        
    elapsed_time = interval

    @property
    def elapsed_seconds(self):
        return self.elapsed_time.total_seconds()
    seconds = elapsed_seconds

    def __exit__(self, *args):
        import streamlit as st
        if self.verbose:
            if self.streamlit:
                st.write(self.text.format(t=self.elapsed_time))
            else: 
                print(self.text.format(t=self.elapsed_time))



def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
            + datetime.timedelta(hours=t.minute // 30))



def roundTime(dt=None, roundTo=60):
   """Round a datetime object to any time lapse in seconds
   dt : datetime.datetime object, default now.
   roundTo : Closest number of seconds to round to, default 1 minute.
   Author: Thierry Husson 2012 - Use it as you want but don't blame me.
   """
   if dt == None : dt = datetime.datetime.now()
   seconds = (dt.replace(tzinfo=None) - dt.min).seconds
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)



def get_current_time():
    return strftime("%m%d%H%M%S", gmtime())

@contextmanager
def timer(name: str) -> None:

    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

