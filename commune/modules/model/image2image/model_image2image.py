import commune as c
import gradio as gr
import requests
import torch
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda:0"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path, torch_dtype=torch.get_default_dtype(),)
pipe = pipe.to(device)
pipe.enable_attention_slicing()


def transform(init_image, textPrompt, strength=0.5, guidance_scale=15):
    init_image = Image.open(init_image).convert("RGB")
    init_image = init_image.resize((768, 512))
    images = pipe(prompt=textPrompt, image=init_image,
                  strength=strength, guidance_scale=guidance_scale).images
    image = images[0]
    return image

class ModelImage2image(c.Module):
    def __init__(self, config=None, **kwargs):
        self.set_config(config, kwargs=kwargs)

    def call(self, x: int = 1, y: int = 2) -> int:
        c.print(self.config.sup)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y

    def gradio(self):
        demo = gr.Interface(
            fn=transform,
            inputs=[gr.Image(type='filepath'), "text",
                    gr.Slider(0, 1), gr.Slider(1, 30)],
            outputs=["image"],
            allow_flagging="never"
        )
        demo.launch()
