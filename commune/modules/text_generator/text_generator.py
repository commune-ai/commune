import commune as c
from text_generation import Client
class TextGenerator(c.Module):
    image = 'text_generator'
    
    def serve(self,  name=None,
                    model = None,
                    tag = None,
                    image=image,
                    num_shard=2, 
                    gpu='all',
                    shm_size='100g',
                    volume=None, 
                    build:bool = True,
                    port=None):
        if model == None:
            model = self.config.model
        name =  image_tag +"_"+ model

        if tag != None:
            name = f'{name}_{tag}'
        

        model_id = self.config.shortcuts.get(model, model)
        
         
        port = c.resolve_port(port)

        if volume == None:
            volume = self.resolve_path('data')
            c.mkdir(volume)
        # if build:
        #     cls.build(tag=tag)
        cmd_args = f'--num-shard {num_shard} --model-id {model_id}'
        cmd = f'docker run -d --gpus {gpu} --shm-size {shm_size} -p {port}:80 -v {volume}:/data --name {name} {image} {cmd_args}'

        output_text = c.cmd(cmd, sudo=True, output_text=True)
        c.print('WHADUP',str(output_text))
        if 'Conflict. The container name' in output_text:
            c.print(f'container {name} already exists, restarting...')
            contianer_id = output_text.split('by container "')[-1].split('". You')[0].strip()
            c.cmd(f'docker rm -f {contianer_id}', sudo=True, verbose=True)
            c.cmd(cmd, sudo=True, verbose=True)


    @classmethod
    def build(cls, image=image):
        cmd = f'sudo docker build -t {image} .'
        c.cmd(cmd, cwd=cls.dirpath(), verbose=True)


    @classmethod
    def namespace(cls):
        output_text = c.cmd('sudo docker ps')
        names = [l.split('  ')[-1].strip() for l in output_text.split('\n')[1:-1]]
        addresses = [l.split('  ')[-2].split('->')[0].strip() for l in output_text.split('\n')[1:-1]]
        namespace = {k:v for k,v in  dict(zip(names, addresses)).items() if k.startswith(cls.image)}
        return namespace

    @classmethod
    def servers(cls):
        return list(cls.namespace().keys())
    @classmethod
    def addresses(cls):
        return list(cls.namespace().values())
    
    @classmethod
    def random_server(cls):
        return c.choice(cls.servers())
    @classmethod
    def random_address(cls):
        return c.choice(cls.addresses())

    @classmethod
    def install(cls):
        c.cmd('pip install -e clients/python/', cwd=cls.dirpath(), verbose=True)


    def generate(self, 
                prompt = 'what is up', 
                max_new_tokens:int=100, 
                model:str = None, 
                **kwargs):
                
        if model == None:
            model = self.random_server()
            address = self.namespace()[model]

        client = Client('http://'+address)
        generated_text = client.generate_stream(prompt, max_new_tokens=max_new_tokens, **kwargs)
        output_text = ''
        for text_obj in generated_text:
            text =  text_obj.token.text
            output_text += text
            c.print(text, end='')

        return output_text

    talk = generate

    