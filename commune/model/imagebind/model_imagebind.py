import commune as c
import logging
import torch
import os
import logging
import torch
from .models import imagebind_model
from .models.imagebind_model import ModalityType, load_module
from .models import lora as LoRA
from .data import *

class Model(c.Module):
    def __init__(self,
        lora = True,
        linear_probing = False,
        device = "cuda",  # "cuda:0" if torch.cuda.is_available() else "cpu"
        load_head_post_proc_finetuned = True
        ):

        logging.basicConfig(level=logging.INFO, force=True)


        assert not (linear_probing and lora), \
                    "Linear probing is a subset of LoRA training procedure for ImageBind. " \
                    "Cannot set both linear_probing=True and lora=True. "

        if lora and not load_head_post_proc_finetuned:
            # Hack: adjust lora_factor to the `max batch size used during training / temperature` to compensate missing norm
            lora_factor = 12 / 0.07
        else:
            # This assumes proper loading of all params but results in shift from original dist in case of LoRA
            lora_factor = 1

        # Instantiate model
        model = imagebind_model.imagebind_huge(pretrained=True)
        if lora:
            model.modality_trunks.update(
                LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
                                                # layer_idxs={ModalityType.TEXT: [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                                #             ModalityType.VISION: [0, 1, 2, 3, 4, 5, 6, 7, 8]},
                                                modality_names=[ModalityType.TEXT, ModalityType.VISION]))

            # Load LoRA params if found
            LoRA.load_lora_modality_trunks(model.modality_trunks,
                                        checkpoint_dir=".checkpoints/lora/550_epochs_lora", postfix="_dreambooth_last")

            if load_head_post_proc_finetuned:
                # Load postprocessors & heads
                load_module(model.modality_postprocessors, module_name="postprocessors",
                            checkpoint_dir=".checkpoints/lora/550_epochs_lora", postfix="_dreambooth_last")
                load_module(model.modality_heads, module_name="heads",
                            checkpoint_dir=".checkpoints/lora/550_epochs_lora", postfix="_dreambooth_last")
        elif linear_probing:
            # Load heads
            load_module(model.modality_heads, module_name="heads",
                        checkpoint_dir="./.checkpoints/lora/500_epochs_lp", postfix="_dreambooth_last")

        model.eval()
        model.to(device)
        self.device = device
        self.model = model
        c.print('Model loaded')


    def forward(self, text='a bird', image=None, audio=None):
        device = self.device
        
        # Load data
        inputs = {}
        if text != None:
            if isinstance(text, str):
                text = [text]
            inputs[ModalityType.TEXT] = load_and_transform_text(text, device, bpe_path = f"{self.dirpath()}/bpe/bpe_simple_vocab_16e6.txt.gz")
        if image != None:
            inputs[ModalityType.VISION] =  load_and_transform_vision_data(image, device, to_tensor=True),

        if audio != None:
            inputs[ModalityType.AUDIO] = load_and_transform_audio_data(audio, device)

        with torch.no_grad():
            embeddings = self.model(inputs)

        return embeddings


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


    @classmethod
    def example(cls, **kwargs):
        c.cmd(f'python3 example.py', cwd=cls.dirpath() )
