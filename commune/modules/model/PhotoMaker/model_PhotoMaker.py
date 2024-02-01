import commune as c
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from ip_adapter import IPAdapterPlusXL
from ip_adapter.custom_pipelines import StableDiffusionXLCustomPipeline

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin" # a experimental version
device = "cuda"

class PhotoMaker(c.Module):
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())
        self.pipe = StableDiffusionXLCustomPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            add_watermarker=False,
        )
        self.ip_model = IPAdapterPlusXL(self.pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config)
        c.print(self.config, 'This is the config, it is a Munch object')

    def image_grid(imgs, rows, cols):
        assert len(imgs) == rows*cols

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols*w, rows*h))
        grid_w, grid_h = grid.size
        
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i%cols*w, i//cols*h))
        return grid
    
    def generate(self, originImg="default.png", promptText="") :
        image = Image.open(originImg)
        image.resize((224, 224))
    
        images = self.ip_model.generate(pil_image=image, num_samples=2, num_inference_steps=30, seed=42, prompt=promptText)
        images[0].save("1.jpg")
        images[1].save("2.jpg")

        c.print("File created. 1.jpg, 2.jpg")
        return
    