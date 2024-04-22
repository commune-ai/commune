import commune as c
from os import path
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image
from ip_adapter import IPAdapterFull
from datetime import datetime

here = path.abspath(path.dirname(__file__))
base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = here + "/models/image_encoder/"
ip_ckpt = here + "/models/ip-adapter-full-face_sd15.bin"
device = "cuda"

class ModelPhotoMaker(c.Module):
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())
                
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        self.vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

        # load SD pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            feature_extractor=None,
            safety_checker=None
        )
        self.ip_model = IPAdapterFull(self.pipe, image_encoder_path, ip_ckpt, device, num_tokens=257)

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    
    def generate(self, imgUrl="assets/images/ai_face.png", promptText="photo of a woman wearing sunglases") :      
        # read image prompt
        image = Image.open(imgUrl)
        image.resize((256, 256))

        # use face as image prompt
        images = self.ip_model.generate(
            pil_image=image, num_samples=4, prompt=promptText,
            scale=0.7, width=512, height=512, num_inference_steps=50, seed=42)
        
        names = []
        for i, image in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            image.save(f"{timestamp}_{i}.png")
            names.append(f"{timestamp}_{i}.png")

        return names