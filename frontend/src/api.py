from fastapi import FastAPI
from pydantic import BaseModel


item_config = {
    'tasks': ['Classification'],
    'user': 'Bobby'
}


modules = {
    'model.transformer' : {
        'tasks': ['Classification']
        
        
    }
}

class Module:
    def __init__(self):
        self.module_name = 'module'

    def predict(self, data: dict):
        return f"Prediction for {self.name}: {data}"


    def list_modules(dict)
    @classmethod
    def api(cls, *args, **kwargs):
        app = FastAPI()
        self = cls(*args, **kwargs)
        
        @app.post("/{fn}")
        async def call_api(fn:str, kwargs: dict = None, args:list = None):
            kwargs = kwargs if kwargs != None else {}
            args = args if args != None else []
            return self(*args, **kwargs)

        return app
    
    
app = Module.api()


    
