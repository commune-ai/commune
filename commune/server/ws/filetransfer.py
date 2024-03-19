import asyncio
import websockets
import commune as c

class FileTransfer(c.Module):
    
    
    def __init__(self,
                 port:int=None,
                 buffer_size:int=-1
                 )
    
        self.set_server(port=port)
        self.set_queue(buffer_size=buffer_size)
    
    @staticmethod
    async def send_file(filename, address):
        async with websockets.connect(address) as websocket:
            with open(filename, 'rb') as file:
                while True:
                    chunk = file.read(1024)
                    if not chunk:
                        break
                    await websocket.send(chunk)
                await websocket.send('END')

    @staticmethod
    async def receive_file(address):
        async with websockets.connect(address) as websocket:
            with open('received_file.txt', 'wb') as file:
                while True:
                    chunk = await websocket.recv()
                    if chunk == 'END':
                        break
                    file.write(chunk)

    async def forward(websocket, path='file_to_send.txt'):
        with open(path, 'rb') as file:
            while True:
                chunk = file.read(1024)
                if not chunk:
                    break
                await websocket.send(chunk)
            await websocket.send('END')

    def set_queue(self, buffer_size:int):
        import queue
        self.queue = queue.Queue(buffer_size)
        

    def set_server(self,port):
        port = self.resolve_port(port)  
        start_server = websockets.serve(self.forward, 'localhost', port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()



if __name__ == "__main__":
    FileTransfer.run()
