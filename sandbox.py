import bittensor as bt

# Query through the foundation endpoint.
print('bro')
print ( bt.prompt( "Heraclitus was a ") )

    
    @classmethod
    def serve(cls, 
              module:Any = None ,
              # name related
              name:str=None, 
              tag:str=None,
              # networking 
              address:str = None,
              ip:str=None, 
              port:int=None ,
              key = None, # key for server's identity
              refresh:bool = True, # refreshes the server's key
              whitelist:List[str] = None, # list of addresses that can connect to the server
              blacklist:List[str] = None, # list of addresses that cannot connect to the server
              wait_for_termination:bool = True, # waits for the server to terminate before returning
              wait_for_server:bool = False, # waits for the server to start before returning
              wait_for_server_timeout:int = 30, # timeout for waiting for the server to start
              wait_for_server_sleep_interval: int = 1, # sleep interval for waiting for the server to start
              verbose:bool = False, # prints out information about the server
              reserve_port:bool = False, # reserves the port for the server
              tag_seperator: str = '::', # seperator for the tag
              remote:bool = True, # runs the server remotely (pm2, ray)
              args:list = None,  # args for the module
              kwargs:dict = None,  # kwargs for the module
              
              ):
        '''
        Servers the module on a specified port
        '''
        kwargs  = kwargs if kwargs else {}
        args = args if args else []
        name = cls.resolve_server_name(module=module, name=name, tag=tag)
        tag = None
        if remote:
            remote_kwargs = cls.locals2kwargs(locals(), merge_kwargs=False)
            remote_kwargs['remote'] = False
            return cls.remote_fn('serve', name=name, kwargs=remote_kwargs, )
        
        if address != None and port == None and ip == None:
            port = int(address.split(':')[-1])
            
        # we want to make sure that the module is loco
        # cls.update(network='local')
    
        module = cls.resolve_module(module)
            
        self = module(*args, **kwargs)

        if whitelist == None:
            whitelist = self.whitelist
        if blacklist == None:
            blacklist = self.blacklist
        
        
            
        '''check if the server exists'''
        c.print(f'Checking if server {name} exists {self}')
        if self.server_exists(name): 
            if refresh:
                if verbose:
                    c.print(f'Stopping server {name}')
                self.kill_server(name)
            else: 
                raise Exception(f'The server {name} already exists on port {existing_server_port}')

        # ensure that the module has a name
        for k in ['module_name', 'module_id', 'my_name', 'el_namo', 'name']:
            if k not in self.__dict__:
                self.__dict__[k] = name

        Server = c.module('module.server')
    
        # ensure the port is free
        if port == None:
            port = cls.free_port(reserve=reserve_port)
            
        server = Server(ip=ip, 
                        port=port,
                        module = self,
                        name= name,
                        whitelist=whitelist,
                        blacklist=blacklist)
        
        # register the server
        self.server_info = server.info
        self.ip = server.ip
        self.port = server.port
        self.address = self.ip_address = self.ip_addy =  server.address
        
        if (not hasattr(self, 'config')) or callable(self.config):
            self.config = cls.munch({})
            
        self.config['info'] = self.info()
        

        # self.set_key(key)
            
        # serve the server
        server.serve(wait_for_termination=wait_for_termination,register=True)
        if wait_for_server:
            cls.wait_for_server(name=module_name, timeout=wait_for_server_timeout, sleep_interval=wait_for_server_sleep_interval)
       