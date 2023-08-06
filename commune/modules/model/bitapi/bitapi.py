import http.client
import json




class BitAPAI(c.Module):
    def __init__(self, conifg=None,  **kwargs):


        self.conn = http.client.HTTPSConnection(self.config.host)
        self.api_key = self.config.api_key

    def forwrd( self, text:str , api_key:str = None, history:list=None, ): 
        payload = [
        {
            "role": "system",
            "content": "You are an AI assistant"
        },
        {
            "role": "user",
            "content": text
        }]

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
        data = res.read()
        return data



