
async def async_read(path, mode='r'):
    import asyncio
    import aiofiles
    async with aiofiles.open(path, mode=mode) as f:
        data = await f.read()
    return data

async def async_write(path, data,  mode ='w'):
    import asyncio
    import aiofiles
    async with aiofiles.open(path, mode=mode) as f:
        await f.write(data)

def sync_wrapper(fn):
    import asyncio
    def wrapper_fn(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return  wrapper_fn