import subprocess
import commune as c
import os

class ModelMetamask(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)

    def run():
        dir_path = os.path.dirname(os.path.realpath(__file__))
        app_path = os.path.join(dir_path, 'app.py')
        subprocess.run(['streamlit', 'run', app_path])