import json

class MunchSerializer:

    def serialize(self, data: dict) -> str:
        return  json.dumps(self.munch2dict(data))

    def deserialize(self, data: bytes) -> 'Munch':
        return self.dict2munch(self.str2dict(data))

    
    def str2dict(self, data:str) -> bytes:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        if isinstance(data, str):
            data = json.loads(data)
        return data

    @classmethod
    def dict2munch(cls, x:dict, recursive:bool=True)-> 'Munch':
        from munch import Munch
        '''
        Turn dictionary into Munch
        '''
        if isinstance(x, dict):
            for k,v in x.items():
                if isinstance(v, dict) and recursive:
                    x[k] = cls.dict2munch(v)
            x = Munch(x)
        return x 

    @classmethod
    def munch2dict(cls, x:'Munch', recursive:bool=True)-> dict:
        from munch import Munch
        '''
        Turn munch object  into dictionary
        '''
        if isinstance(x, Munch):
            x = dict(x)
            for k,v in x.items():
                if isinstance(v, Munch) and recursive:
                    x[k] = cls.munch2dict(v)
        return x 
        

    def dict2str(self, data:dict) -> bytes:
        return

