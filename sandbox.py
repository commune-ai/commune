async def gen():
    for i in range(3):
        print('yielding')
        yield i

g = gen()
async def f(x=1, g=g):
    return await g.__anext__()
import asyncio
for i in range(10):
    print(asyncio.run(f(g=g)))
    
