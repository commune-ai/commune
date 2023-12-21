import subprocess
import commune as c
import os

class ModelMetamask(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)

    def run(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        app_path = os.path.join(dir_path, 'app.py')
        backend_api_path = os.path.join(dir_path, 'backend_api.py')
        subprocess.Popen(['streamlit', 'run', app_path])
        subprocess.Popen(['python3', backend_api_path])