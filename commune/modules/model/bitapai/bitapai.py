import http.client
import json
import commune as c




class BitAPAI(c.Module):
    def __init__(self,  config=None,  **kwargs):
        config = self.set_config(config=config, kwargs=kwargs)
        self.conn = http.client.HTTPSConnection(self.config.host)
        self.set_api_key(api_key=config.api_key)

    def set_api_key(self, api_key:str):
        self.api_key = api_key

    def build_payload(self, text: str, history:list):
        payload = [
        {
            "role": "system",
            "content": "You are an AI assistant"
        },
        {
            "role": "user",
            "content": text
        }
        ]

        if history is not None:
            assert isinstance(history, list)
            assert len(history) > 0
            assert all([isinstance(i, dict) for i in history])
            payload = payload[:1] +  history + payload[1:]
        return payload
    
    
    def forward( self, text:str , api_key:str = None, history:list=None): 
        api_key = api_key if api_key != None else self.api_key
        # build payload
        payload = self.build_payload(text=text, history=history)
        payload = json.dumps(payload)
        headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': api_key
        }
        self.conn.request("POST", "/text", payload, headers)
        res = self.conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))
        if 'assistant' not in data:
            return data
        return data['assistant']
    
    
    talk = generate = forward
    
    def test(self):
        return self.forward("hello")



