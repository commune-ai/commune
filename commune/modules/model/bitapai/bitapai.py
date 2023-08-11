import http.client
import json
import commune as c




class BitAPAI(c.Module):
    def __init__(self, config=None,  **kwargs):
        self.set_config(config=config, kwargs=kwargs)
        self.conn = http.client.HTTPSConnection(self.config.host)
        self.api_key = self.config.api_key

    def forward( self, text:str , api_key:str = None, history:list=None, ): 
        api_key = api_key or self.api_key
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

        payload = json.dumps(payload)
        headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': api_key or self.api_key
        }
        self.conn.request("POST", "/v1/conversation", 
                          payload, 
                          headers)
        res = self.conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))
        return data['assistant']
    
    talk = generate = forward
    
    def test(self):
        return self.forward("hello")



