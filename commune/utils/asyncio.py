import asyncio
import aiofiles
async def async_read(path, mode='r'):
    async with aiofiles.open(path, mode=mode) as f:
        data = await f.read()
    return data

async def async_write(path, data,  mode ='w'):
    async with aiofiles.open(path, mode=mode) as f:
        await f.write(data)

def sync_wrapper(fn):
    
    loop = get_event_loop()
    def wrapper_fn(*args, **kwargs):
        return loop.run_until_complete(fn(*args, **kwargs))
    return  wrapper_fn

def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop