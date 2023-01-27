import datetime
import time

def get_current_time():
    return time.strftime("%m%d%H%M%S", time.gmtime())


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



class timer:
    def __init__(self, start=True):
        if start:
            self.start()
    
    def start(self):
        self.start_time = time.time()
        return self.start_time
    
    @property
    def seconds(self) -> int:
        return time.time() - self.start_time
    def stop(self) -> int:
        time_passed  = self.seconds
        return time_passed
        
        
    def __enter__(self):
        self.start()
        return self
    def __exit__(self, *args):
        self.stop()


Timer = timer
