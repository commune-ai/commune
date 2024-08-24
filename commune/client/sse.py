

import json

class ClientSSE:

    def iter_over_async(self, ait):
        # helper async fn that just gets the next element
        # from the async iterator
        def get_next():
            try:
                obj = self.loop.run_until_complete(ait.__anext__())
                return obj
            except StopAsyncIteration:
                return 'done'
        # actual sync iterator (implemented using a generator)
        while True:
            obj = get_next() 
            if obj == 'done':
                break
            yield obj

    async def stream_generator(self, response):
        async for line in response.content:
            event =  self.process_stream_line(line)
            if event == '':
                continue
            yield event


    def stream(self,response ):
        return self.iter_over_async(self.stream_generator(response))

    
    def process_stream_line(self, line , stream_prefix=None):
        stream_prefix = stream_prefix or self.stream_prefix
        event_data = line.decode('utf-8')
        if event_data.startswith(stream_prefix):
            event_data = event_data[len(stream_prefix):] 
        event_data = event_data.strip() # remove leading and trailing whitespaces
        if event_data == "": # skip empty lines if the event data is empty
            return ''
        if isinstance(event_data, str):
            if event_data.startswith('{') and event_data.endswith('}') and 'data' in event_data:
                event_data = json.loads(event_data)['data']
        return event_data