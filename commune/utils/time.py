
def time() -> float:
    import time
    return time.time()
    
def get_current_time():
    from time import strftime
    from time import gmtime
    return strftime("%m%d%H%M%S", gmtime())
def roundTime(dt=None, roundTo=60):
   """Round a datetime object to any time lapse in seconds
   dt : datetime.datetime object, default now.
   roundTo : Closest number of seconds to round to, default 1 minute.
   Author: Thierry Husson 2012 - Use it as you want but don't blame me.
   """
   import datatime
   if dt == None : dt = datetime.datetime.now()
   seconds = (dt.replace(tzinfo=None) - dt.min).seconds
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)

def get_current_time() -> str:
    import time
    return time.strftime("%m%d%H%M%S", time.gmtime())

def isoformat2datetime(isoformat:str) -> 'datetime.datetime':
    import datetime
    
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
    import datetime
    assert len(kwargs) == 1
    supported_modes = ['hours', 'seconds', 'minutes', 'days']
    mode = list(kwargs.keys())[0]
    assert mode in supported_modes, f'return type should in {supported_modes} but you put {mode}'
    
    current_timestamp = datetime.datetime.utcnow()
    timetamp_delta  =  current_timestamp.timestamp() -  ( current_timestamp- datetime.timedelta(**kwargs)).timestamp()
    return timetamp_delta



def hour_rounder(t):
    import datetime
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
            + datetime.timedelta(hours=t.minute // 30))

def timestamp() -> float:
    return time()

def sleep(seconds:float) -> None:
    import time
    time.sleep(seconds)
    return None

def datetime():
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S")

def time2datetime(t:float):
    import datetime
    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d_%H:%M:%S")
