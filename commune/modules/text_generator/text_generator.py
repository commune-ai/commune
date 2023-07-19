import commune as c
class TextGenerator(c.Module):
    image = 'text_generator'
    
    def serve(self,tag = None,
                    num_shard=2, 
                    gpus='all',
                    shm_size='100g',
                    volume=None, 
                    build:bool = True,
                    model = None,
                    sudo = True,
                    port=None):
        if model == None:
            model = self.config.model
        name =  self.image +"_"+ model

        if tag != None:
            name = f'{name}_{tag}'
        

        model_id = self.config.shortcuts.get(model, model)
        
        if port == None:
            port = c.resolve_port(port)

        if volume == None:
            volume = self.resolve_path('data')
            c.mkdir(volume)
        # if build:
        #     self.build(tag=tag)
        cmd_args = f'--num-shard {num_shard} --model-id {model_id}'
        cmd = f'docker run -d --gpus \'"device={gpus}"\' --shm-size {shm_size} -p {port}:80 -v {volume}:/data --name {name} {self.image} {cmd_args}'

        
        output_text = c.cmd(cmd, sudo=sudo, output_text=True)


        if 'Conflict. The container name' in output_text:
            c.print(f'container {name} already exists, restarting...')
            contianer_id = output_text.split('by container "')[-1].split('". You')[0].strip()
            c.cmd(f'docker rm -f {contianer_id}', sudo=sudo, verbose=True)
            c.cmd(cmd, sudo=sudo, verbose=True)
    # def fleet(self, num_shards = 2, buffer=5_000_000_000, **kwargs):
    #     model_size = c.get_model_size(self.config.model)
    #     model_shard_size = (model_size // num_shards ) + buffer
    #     max_gpu_memory = c.max_gpu_memory(model_size)
    #     c.print(max_gpu_memory, model_shard_size)


    #     c.print(gpus)


        
        

    def build(self, ):
        cmd = f'docker build -t {self.image} .'
        c.cmd(cmd, cwd=self.dirpath(), verbose=True)

    def logs(self, name, sudo=False):
        return c.cmd(f'docker logs {name}', cwd=self.dirpath())


    def namespace(self, external_ip = True ):
        output_text = c.cmd('docker ps', )
        names = [l.split('  ')[-1].strip() for l in output_text.split('\n')[1:-1]]
        addresses = [l.split('  ')[-2].split('->')[0].strip() for l in output_text.split('\n')[1:-1]]
        namespace = {k:v for k,v in  dict(zip(names, addresses)).items() if k.startswith(self.image)}
        if external_ip:
            namespace = {k:v.replace(c.default_ip, c.external_ip()) for k,v in namespace.items()}
        return namespace

    
    def servers(self):
        return list(self.namespace().keys())
    
    def addresses(self):
        return list(self.namespace().values())
    
    def random_server(self):
        return c.choice(self.servers())
    
    def random_address(self):
        return c.choice(self.addresses())

    
    def install(self):
        c.cmd('pip install -e clients/python/', cwd=self.dirpath(), verbose=True)


    @classmethod
    def generate(cls, 
                prompt = 'what is up, how is it going bro what are you saying?', 
                model:str = None,
                max_new_tokens:int=100, 
                timeout = 6,
                **kwargs):

        self = cls()
        namespace = self.namespace()

        if model == None and address==None:
            assert len(namespace) > 0, f'No models found, please run {self.image}.serve() to start a model server'
            address = self.random_address()
            
        address = namespace.get(model, model)

        if not address.startswith('http://'):
            address = 'http://'+address
            
        from text_generation import Client
        client = Client(address)
        generated_text = client.generate_stream(prompt, max_new_tokens=max_new_tokens, **kwargs)
        output_text = ''

        start_time = c.time()
        for text_obj in generated_text:
            if c.time() - start_time > timeout:
                break
            text =  text_obj.token.text
            output_text += text
            c.print(text, end='')


        return output_text

    talk = generate

    