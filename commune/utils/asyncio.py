import asyncio

async def async_read(path, mode='r'):
    import aiofiles
    async with aiofiles.open(path, mode=mode) as f:
        data = await f.read()
    return data

def get_new_event_loop(nest_asyncio:bool = False):
    if nest_asyncio:
        set_nest_asyncio()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop

def sync_wrapper(fn):
    
    def wrapper_fn(*args, **kwargs):
        if 'loop'  in kwargs:
            loop = kwargs['loop']
        else:
            loop = get_event_loop()
        return loop.run_until_complete(fn(*args, **kwargs))
    return  wrapper_fn

def new_event_loop(nest_asyncio:bool = True) -> 'asyncio.AbstractEventLoop':
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if nest_asyncio:
        set_nest_asyncio()
    return loop

def set_event_loop(self, loop=None, new_loop:bool = False) -> 'asyncio.AbstractEventLoop':
    import asyncio
    try:
        if new_loop:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            loop = loop if loop else asyncio.get_event_loop()
    except RuntimeError as e:
        self.new_event_loop()
        
    self.loop = loop
    return self.loop

def get_event_loop(nest_asyncio:bool = True) -> 'asyncio.AbstractEventLoop':
    try:
        loop = asyncio.get_event_loop()
    except Exception as e:
        loop = new_event_loop(nest_asyncio=nest_asyncio)
    return loop

def set_nest_asyncio():
    import nest_asyncio
    nest_asyncio.apply()
    

def gather(jobs:list, timeout:int = 20, loop=None)-> list:

    if loop == None:
        loop = get_event_loop()

    if not isinstance(jobs, list):
        singleton = True
        jobs = [jobs]
    else:
        singleton = False

    assert isinstance(jobs, list) and len(jobs) > 0, f'Invalid jobs: {jobs}'
    # determine if we are using asyncio or multiprocessing

    # wait until they finish, and if they dont, give them none

    # return the futures that done timeout or not
    async def wait_for(future, timeout):
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            result = {'error': f'TimeoutError: {timeout} seconds'}
        return result
    
    jobs = [wait_for(job, timeout=timeout) for job in jobs]
    future = asyncio.gather(*jobs)
    results = loop.run_until_complete(future)

    if singleton:
        return results[0]
    return results


