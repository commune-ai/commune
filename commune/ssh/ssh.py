import commune as c
import os
class SSH(c.Module):
    key_path = os.path.expanduser('~/.ssh/id_ed25519')
    public_key_path = key_path + '.pub'
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())

    def public_key(self):
        return c.get_text(self.public_key_path).split('\n')[0].split(' ')[1]


    def create_key(self, key:str = None):
        if key == None:
            key = self.key_path
        return c.cmd(f'ssh-keygen -o -a 100 -t ed25519 -f ~/.ssh/id_ed25519')
    
