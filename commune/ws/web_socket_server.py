import threading
import asyncio
import websockets

import commune

class WSServer(commune.Module):
    threads = {}
    def __init__(self, config = None, **kwargs):
        config = self.set_config(config,kwargs=kwargs)
        self.start_server(name='main',ip=config.ip, port=config.port)
    
    @staticmethod
    async def handler(websocket):
        while True:
            try:
                message = await websocket.recv()
            except websockets.ConnectionClosedOK:
                break
            print(message)


    async def async_run_server(self, ip, port):
        async with websockets.serve(self.handler, ip, port):
            await asyncio.Future()  # run forever

    def run_server(self, ip, port):
        loop =  self.new_event_loop()
        loop.run_until_complete(self.async_run_server(ip,port))

    @staticmethod
    def is_address(address):
        conditions = [
            isinstance(address, str),
            len(address.split(':')) == 2,
            address.split(':')[1].isdigit(),   
        ]
        
        return all(conditions)
        
        
    # def rm_server(self, name):
    #     self.name2address = self.get(f'name2address', {})

    #     self.threads[name].join()
    #      self.threads[name]
    #     self.name2address.pop(name)
        
    #     self.put(f'name2address', self.name2address)


    def start_server(self, name, ip='0.0.0.0',port=None):
        self.name2address = self.get(f'name2address', {})
        port = self.free_port() if port is None else port
        
        thread = threading.Thread(target=self.run_server, kwargs=dict(ip=ip, port=port))
        thread.deamon = self.config.deamon  # Set the daemon attribute to True
        thread.start()
        self.name2address[name] = f'{ip}:{port}'
        self.threads[name] = thread
        self.put(f'name2address', self.name2address[name])
        
    # def run_server(self, ip:str = None, port:int = None):
    #     asyncio.set_event_loop(asyncio.new_event_loop())
    #     server = websockets.serve(self.handle_client, ip, port)
    #     asyncio.get_event_loop().run_until_complete(server)
    #     asyncio.get_event_loop().run_forever()

        
    @classmethod
    def test(cls):
        self = cls()
        
        
WSServer()