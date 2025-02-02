import commune as c
from typing import *
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn
import os
import json
import asyncio
import os
import commune as c
from typing import *
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn
import os
import json
import asyncio


class Gate:

    def __init__(self, module, network='subspace', max_network_age=60, history_path=None, max_user_history_age=60):
        self.module = module
        self.max_network_age = max_network_age
        self.sync_network(network)
        self.history_path = history_path or self.resolve_path('gate')
        self.max_user_history_age = max_user_history_age


    @classmethod
    def resolve_path(cls, path):
        return  c.storage_path + '/server.gate/' + path

    state = {}
    def sync_network(self, network=None):
        self.network = network or self.network
        self.network_path = self.resolve_path(f'networks/{self.network}/state.json')
        self.address2key =  c.address2key()
        c.print(f'Network(network={self.network} path={self.network_path})')
        self.state = c.get(self.network_path, {}, max_age=self.max_network_age)
        if self.state == {}:
            def sync():
                self.network_module = c.module(self.network)()
                self.state = self.network_module.state()
            c.thread(sync)
        return {'network':self.network}

    def sync_loop(self):
        c.sleep(self.max_network_age/2)
        while True:
            try:
                r = self.sync_network()
            except Exception as e:
                r = c.detailed_error(e)
                c.print('Error in sync_loop -->', r, color='red')
            c.sleep(self.max_network_age)


    def is_admin(self, key_address):
        return c.is_admin(key_address)

    def get_user_role(self, key_address):
        if c.is_admin(key_address):
            return 'admin'
        if key_address == self.module.key.ss58_address:
            return 'owner'
        if key_address in self.address2key:
            return 'local'
        return 'stake'

    def forward(self, 
                fn:str, 
                params:dict,  
                headers:dict, 
                multipliers : Dict[str, float] = {'stake': 1, 'stake_to': 1,'stake_from': 1}, 
                rates : Dict[str, int]= {'local': 10000, 'owner': 10000, 'admin': 10000}, # the maximum rate  ):
                max_request_staleness : int = 4 # (in seconds) the time it takes for the request to be too old
            ) -> bool:
            role = self.get_user_role(headers['key'])
            if role == 'admin':
                return True
            if self.module.free: 
                return True
            stake = 0
            assert fn in self.module.fns , f"Function {fn} not in endpoints={self.module.fns}"
            request_staleness = c.time() - float(headers['time'])
            assert  request_staleness < max_request_staleness, f"Request is too old ({request_staleness}s > {max_request_staleness}s (MAX)" 
            auth = {'params': params, 'time': str(headers['time'])}
            assert c.verify(auth=auth,signature=headers['signature'], address=headers['key']), 'Invalid signature'
            role = self.get_user_role(headers['key'])
            if role in rates:
                rate_limit = rates[role]
            else:
                stake = self.state['stake'].get(headers['key'], 0) * self.multipliers['stake']
                stake_to = (sum(self.state['stake_to'].get(headers['key'], {}).values())) * multipliers['stake_to']
                stake_from = self.state['stake_from'].get(self.module.key.ss58_address, {}).get(headers['key'], 0) * multipliers['stake_from']
                stake = stake + stake_to + stake_from
                raet_limit = rates['stake'] / self.module.fn2cost.get(fn, 1)
                rate_limit =  min(raet_limit, rates['max'])
            rate = self.call_rate(headers['key'])
            assert rate <= rate_limit, f'RateLimitExceeded({rate}>{rate_limit})'     
            return {'rate': rate, 
                    'rate_limit': rate_limit, 
                    'cost': self.module.fn2cost.get(fn, 1)
                    }

    
    def user_call_path2latency(self, key_address):
        user_paths = self.call_paths(key_address)
        t1 = c.time()
        user_path2time = {p: t1 - self.path2time(p) for p in user_paths}
        return user_path2time

    def get_call_data_path(self, key_address):
        return self.history_path + '/' + key_address

    def call_rate(self, key_address, max_age = 60):
        path2latency = self.user_call_path2latency(key_address)
        for path, latency  in path2latency.items():
            if latency > self.max_user_history_age:
                c.print(f'RemovingUserPath(path={path} latency(s)=({latency}/{self.max_user_history_age})')
                if os.path.exists(path):
                    os.remove(path)
        return len(self.call_paths(key_address))

    def user_history(self, key_address, stream=False):
        call_paths = self.call_paths(key_address)
        if stream:
            def stream_fn():
                for p in call_paths:
                    yield c.get(p)
            return stream_fn()
        else:
            return [c.get(call_path) for call_path in call_paths]
        
    def user2fn2calls(self):
        user2fn2calls = {}
        for user in self.users():
            user2fn2calls[user] = {}
            for user_history in self.user_history(user):
                fn = user_history['fn']
                user2fn2calls[user][fn] = user2fn2calls[user].get(fn, 0) + 1
        return user2fn2calls

    def call_paths(self, key_address ):
        user_paths = c.glob(self.get_call_data_path(key_address))
        return sorted(user_paths, key=self.path2time)

    def users(self):
        return os.listdir(self.history_path)

    def history(self, module=None , simple=True):
        module = module or self.module.name
        all_history = {}
        users = self.users()
        for user in users:
            all_history[user] = self.user_history(user)
            if simple:
                all_history[user].pop('output')
        return all_history
    @classmethod
    def all_history(cls, module=None):
        self = cls(module=module, run_api=False)
        all_history = {}
        return all_history

    def path2time(self, path:str) -> float:
        try:
            x = float(path.split('/')[-1].split('.')[0])
        except Exception as e:
            x = 0
        return x
        return Middleware


    def set_middleware(self, app, max_bytes=1000000):
    
        from starlette.middleware.base import BaseHTTPMiddleware
        class Middleware(BaseHTTPMiddleware):
            def __init__(self, app, max_bytes: int = 1000000):
                super().__init__(app)
                self.max_bytes = max_bytes
            async def dispatch(self, request: Request, call_next):
                content_length = request.headers.get('content-length')
                if content_length:
                    if int(content_length) > self.max_bytes:
                        return JSONResponse(status_code=413, content={"error": "Request too large"})
                body = await request.body()
                if len(body) > self.max_bytes:
                    return JSONResponse(status_code=413, content={"error": "Request too large"})
                response = await call_next(request)
                return response

        app.add_middleware(Middleware, max_bytes=max_bytes)
        app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
        return app


    def save_data(self, data):
        call_data_path = self.get_call_data_path(f'{data["key"]}/{data["fn"]}/{c.time()}.json') 
        c.put(call_data_path, data)
        return call_data_path