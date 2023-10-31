import http.client
import json
import commune as c




class BitAPAI(c.Module):
    
    whitelist = ['forward', 'chat', 'ask', 'generate']

    def __init__(self, api_key:str = None, host='api.bitapai.io', cache_key:bool = True):
        config = self.set_config(kwargs=locals())
        self.conn = http.client.HTTPSConnection(self.config.host)
        self.set_api_key(api_key=config.api_key, cache=config.cache_key)
        
    def set_api_key(self, api_key:str, cache:bool = True):
        if api_key == None:
            api_key = self.get_api_key()

        
        self.api_key = api_key
        if cache:
            self.add_api_key(api_key)

        assert isinstance(api_key, str)

            
    
    def forward( self, 
                text:str ,
                # api default is 20, I would not go less than 10 with current network conditions
                # larger number = higher query spread across top miners but slightly longer query time
                 count:int = 300,
                 # changed to False, I assume you only want a single repsonse so it will return a single random from the pool of valid responses
                 return_all:bool = False,
                 # added exclude_unavailable to ensure no empty responses are returned
                 exclude_unavailable:bool = True,
                 uids: list = None,
                 api_key:str = None, 
                 history:list=None) -> str: 
        api_key = api_key if api_key != None else self.api_key
        # build payload


        payload =  {
            'messages': 
                    [
                    {
                        "role": "system",
                        "content": "You are an AI assistant"
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                    ],
            "count": count,
            "return_all": return_all,
            # added return all here
            "exclude_unavailable": exclude_unavailable

        }
        if uids is not None:
            payload['uids'] = uids

        if history is not None:
            assert isinstance(history, list)
            assert len(history) > 0
            assert all([isinstance(i, dict) for i in history])
            payload = payload[:1] +  history + payload[1:]


        payload = json.dumps(payload)
        headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': api_key
        }

        
        self.conn.request("POST", "/text", payload, headers)


        res = self.conn.getresponse()
        data = res.read().decode("utf-8")
        data = json.loads(data)
        c.print(len(data['choices']))

        return data['choices'][0]['message']['content']
    
    
    talk = generate = forward
    
    def test(self):
        return self.forward("hello")


    ## API MANAGEMENT ##

    @classmethod
    def add_api_key(cls, api_key:str):
        assert isinstance(api_key, str)
        api_keys = cls.get('api_keys', [])
        api_keys.append(api_key)
        api_keys = list(set(api_keys))
        cls.put('api_keys', api_keys)
        return {'api_keys': api_keys}


    @classmethod
    def rm_api_key(cls, api_key:str):
        assert isinstance(api_key, str)
        api_keys = cls.get('api_keys', [])
        for i in range(len(api_keys)):
            if api_key == api_keys[i]:
                api_keys.pop(i)
                break   

        cls.put('api_keys', api_keys)
        return {'api_keys': api_keys}


    @classmethod
    def get_api_key(cls):
        api_keys = cls.api_keys()
        if len(api_keys) == 0:
            return None
        else:
            return c.choice(api_keys)

    @classmethod
    def api_keys(cls):
        return cls.get('api_keys', [])

 