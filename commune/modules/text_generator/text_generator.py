import commune as c

class TextGenerator(c.Module):
    docker_image_tag = 'text_generator'
    @classmethod
    def run(cls, tag=docker_image_tag, gput='all', shm_size='100g', volume:str=None, port:str=8080):
        volume = volume or cls.resolve_path('data')
        c.cmd(f'docker run --gpus all --shm-size 100g -p {port}:80 -v {volume}:/data {docker_image_tag}')

    @classmethod
    def build(cls, tag=docker_image_tag):
        cmd = f'sudo docker build -t {tag} .'
        c.cmd(cmd, cwd=cls.dirpath(), verbose=True)

    @classmethod
    def build(cls, tag=docker_image_tag):
        c.cmd(f'sudo docker build -t {tag} .', cwd=cls.dirpath(), verbose=True)
        
    