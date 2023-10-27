import commune as c
class TextGenerator(c.Module):
    image = 'text_generator'


    def fleet(self, model = 'vicuna.7b', n=None):
        if n == None:
            n = c.num_gpus()
        free_gpu_memory = c.free_gpu_memory()
        fleet_gpus = {}
        if isinstance(model, str):
            models = [model]*n
        elif isinstance(model, list):
            models = model
        else:
            raise ValueError(f'model must be a str or list, got {type(model)}')


        for i, model in enumerate(models):
            model_gpu_memory = c.model_max_gpu_memory(model, free_gpu_memory=free_gpu_memory)
            model_gpus = list(model_gpu_memory.keys())
            for k,v in model_gpu_memory.items():
                free_gpu_memory[k] -= v
                if free_gpu_memory[k] < 0:
                    free_gpu_memory[k] = 0

            c.print(f'model {i} gpus: {model_gpus}')


            self.deploy_server(model, gpus=model_gpus, tag=str(i))

    
    def deploy_server(self, model :str = None,
                    tag: str = None,
                    num_shard:int=None, 
                    gpus:list=None,
                    shm_size : str='100g',
                    volume:str = 'data',
                    build:bool = False,
                    max_shard_ratio = 0.5,
                    refresh:bool = False,
                    sudo = False,
                    port=None):

        if model == None:
            model = self.config.model
        if tag != None:
            tag = str(tag)
        name =  (self.image +"_"+ model) + ('_'+tag if tag  else '')

        c.print('name: ', name, model)



        if self.server_exists(name) and refresh == False:
            c.print(f'{name} already exists')
            return

        if build:
            self.build()

        if gpus == None:
            gpus = c.model_max_gpus(model)
        
        num_shard = len(gpus)
        gpus = ','.join(map(str, gpus))

        c.print(f'gpus: {gpus}')
        
        model_id = self.config.shortcuts.get(model, model)
        if port == None:
            port = c.resolve_port(port)

        volume = self.resolve_path(volume)
        if not c.exists(volume):
            c.mkdir(volume)

        cmd_args = f'--num-shard {num_shard} --model-id {model_id}'
        cmd = f'docker run -d --gpus \'"device={gpus}"\' --shm-size {shm_size} -p {port}:80 -v {volume}:/data --name {name} {self.image} {cmd_args}'

        c.print(cmd, 'BROOOO')
        output_text = c.cmd(cmd, sudo=sudo, output_text=True)
        if 'Conflict. The container name' in output_text:
            c.print(f'container {name} already exists, restarting...')
            contianer_id = output_text.split('by container "')[-1].split('". You')[0].strip()
            c.cmd(f'docker rm -f {contianer_id}', sudo=sudo, verbose=True)
            c.cmd(cmd, sudo=sudo, verbose=True)

        else: 
            c.print(output_text)


        self.update()
        

    def build(self ):
        cmd = f'docker build -t {self.image} .'
        c.cmd(cmd, cwd=self.dirpath(), env={'DOCKER_BUILDKIT':'1'},verbose=True)

    def logs(self, name, sudo=False, follow=False):
        return c.cmd(f'docker {"-f" if follow else ""} logs text_generator_{name}', cwd=self.dirpath(), verbose=False)

    def server2logs(self, sudo=False):
        return {k:self.logs(k, sudo=sudo, follow=False)[-50:] for k in self.servers()}

    def update(self):
        return self.namespace(load=False, save=True)
        

    def namespace(self, external_ip = True , load=True, save=False ):

        if load and save == False:
            namespace = self.get('namespace', {})
            if len(namespace) > 0:
                return namespace

        output_text = c.cmd('docker ps', verbose=False)
        names = [l.split('  ')[-1].strip() for l in output_text.split('\n')[1:-1]]
        addresses = [l.split('  ')[-2].split('->')[0].strip() for l in output_text.split('\n')[1:-1]]
        namespace = {k:v for k,v in  dict(zip(names, addresses)).items() if k.startswith(self.image)}
        if external_ip:
            namespace = {k.replace(self.image+'_', ''):v.replace(c.default_ip, c.external_ip()) for k,v in namespace.items()}
        if save:
            self.put('namespace', namespace)

        return namespace

    
    def servers(self):
        return list(self.namespace().keys())
    models = servers
    
    def server_exists(self, model):
        servers = self.servers()
        return model in servers

    model_exists = server_exists


    def kill(self, model):
        assert self.server_exists(model), f'{model} does not exist'
        c.cmd(f'docker rm -f {self.image}_{model}', verbose=True)
        self.update()
        assert not self.server_exists(model), f'failed to kill {model}'
        return {'status':'killed', 'model':model}



    def purge(self):
        for model in self.servers():
            if self.server_exists(model):
                self.kill(model)

    def addresses(self):
        return list(self.namespace().values())
    
    def random_server(self):
        return c.choice(self.servers())
    
    def random_address(self):
        return c.choice(self.addresses())

    
    def install(self):
        c.cmd('pip install -e clients/python/', cwd=self.dirpath(), verbose=True)


    @classmethod
    def generate_stream(cls, 
                prompt = 'what is up, how is it going bro what are you saying?', 
                model:str = None,
                max_new_tokens:int=100, 
                trials = 4,
                timeout = 6,
                **kwargs):

        self = cls()
        namespace = self.namespace()

        if model == None:
            assert len(namespace) > 0, f'No models found, please run {self.image}.serve() to start a model server'
            model = self.random_server()
            
        address = namespace.get(model, model)

        if not address.startswith('http://'):
            address = 'http://'+address

        c.print(f'address: {address}')   
        from text_generation import Client

        try:
            client = Client(address)
            generated_text = client.generate_stream(prompt, max_new_tokens=max_new_tokens, **kwargs)
        except Exception as e:
            c.print(f'error generating text, retrying -> {trials} left...')
            trials -= 1
            if trials > 0:
                return cls.generate(prompt=prompt, 
                            model=model, 
                            max_new_tokens=max_new_tokens, 
                            trials=trials, 
                            timeout=timeout, 
                            **kwargs)
            else:
                raise Exception(f'error generating text, retrying -> {trials} left...')
        output_text = ''

        start_time = c.time()
        for text_obj in generated_text:
            if c.time() - start_time > timeout:
                break
            text =  text_obj.token.text
            output_text += text
            c.print(text, end='')


        return output_text


    @classmethod
    def generate(cls, 
                prompt = 'what is up, how is it going bro what are you saying? A: ', 
                model:str = None,
                max_new_tokens:int=256, 
                timeout = 6,
                return_dict:bool = False,

                **kwargs):

        self = cls()
        namespace = self.namespace()

        if model == None:
            assert len(namespace) > 0, f'No models found, please run {self.image}.serve() to start a model server'
            model = self.random_server()
            
        address = namespace.get(model, model)

        if not address.startswith('http://'):
            address = 'http://'+address

        c.print(f'address: {address}')   
        from text_generation import Client

        # try:
        client = Client(address)
        t = c.time()
        output_text = client.generate(prompt, max_new_tokens=max_new_tokens, **kwargs).generated_text
        stats = {
            'time': c.time() - t,
            'model': model,
            'output_tokens': len(c.tokenize(c.copy(output_text))),
        }
        stats['tokens_per_second'] = stats['output_tokens'] / stats['time']
        c.print(output_text)
        if return_dict:
            return {'text' : output_text, **stats}
        else:
            return output_text

    talk = generate


    @classmethod
    def client_fleet(cls, n=10):
        for i in range(n):
            cls.serve(name=f'model.textgen.{i}')


    