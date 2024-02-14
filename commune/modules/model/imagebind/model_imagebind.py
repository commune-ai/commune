import commune as c
import logging
import torch
import os

class Model(c.Module):
    def __init__(self,**kwargs):
        self.set_model(**kwargs)
    def set_model(self, **kwargs):
        from .models.imagebind_model import ImageBindModel
        kwargs.pop('tag', None)
        self.model = ImageBindModel(**kwargs)
        self.load_module(self.module)
        return {'succes': True, 'msg': f'model set'}

    @staticmethod
    def save_module(self, module_name: str = "",
                    checkpoint_dir: str = "./.checkpoints/full", postfix: str = "_last",
                    extension: str = "pth"):
    
        try:
            torch.save(self.model.state_dict(),
                    os.path.join(checkpoint_dir, f"imagebind-{module_name}{postfix}.{extension}"))
            logging.info(f"Saved parameters for module {module_name} to {checkpoint_dir}.")
        except FileNotFoundError:
            logging.warning(f"Could not save module parameters for {module_name} to {checkpoint_dir}.")

    def load_module(self,
                    module_name: str = "default",
                    checkpoint_dir: str = "./.checkpoints/full", 
                    postfix: str = "_last",
                    extension: str = "pth"):
        try:
            self.model.load_state_dict(torch.load(
                    os.path.join(checkpoint_dir, f"imagebind-{module_name}{postfix}.{extension}")), strict=False)
            logging.info(f"Loaded parameters for module {module_name} from {checkpoint_dir}.")
        except FileNotFoundError:
            logging.warning(f"Could not load module parameters for {module_name} from {checkpoint_dir}.")


    @classmethod
    def install(cls, **kwargs):
        path =  cls.dirpath() + '/requirements.txt'
        c.cmd(f'pip3 install -r {path}')