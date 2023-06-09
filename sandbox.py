import asyncio
import websockets
import json
import commune as c

class WebSocketServerModule(c.Module):
    def __init__(self, **kwargs):
        config = self.set_config(config=kwargs)
        self.uri = config['uri']
        self.channel = config['channel']
        self.remote_uri = config.get('remote_uri', None)  # URI of the other server
        self.queue = asyncio.Queue()

    async def server(self, websocket, path):
        async for message in websocket:
            await self.queue.put(message)

    async def process(self):
        while True:
            message = await self.queue.get()
            message_data = json.loads(message)
            print(f"Received: {message_data}")

            # If a remote_uri is set, send the received message to the other server
            if self.remote_uri:
                await self.send_to_remote(json.dumps(message_data))

    async def send_to_remote(self, message):
        async with websockets.connect(self.remote_uri) as websocket:
            await websocket.send(message)
            print(f"Sent: {message}")

    def run(self):
        start_server = websockets.serve(self.server, self.uri)

        asyncio.ensure_future(self.process())
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

if __name__ == '__main__':
    server1 = WebSocketServerModule(uri="ws://localhost:8765", channel="test_channel", remote_uri="ws://localhost:8766")
    server1.run()

    server2 = WebSocketServerModule(uri="ws://localhost:8766", channel="test_channel", remote_uri="ws://localhost:8765")
    server2.run()
