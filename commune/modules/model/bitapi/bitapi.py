import http.client
import json




class BitAPI(c.Module):
    def __init__(slconifg=None,  **kwargs):
        


        self.conn = http.client.HTTPSConnection("api.bitapai.io")

    def forwrd( text , )
        payload = json.dumps([
        {
            "role": "system",
            "content": "You are an AI assistant"
        },
        {
            "role": "user",
            "content": text
        }])
        headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': self.api_key
        }
        conn.request("POST", "/v1/conversation", payload, headers)
        res = conn.getresponse()
        data = res.read()
        return 


