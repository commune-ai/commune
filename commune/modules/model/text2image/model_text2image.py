import commune as c
from diffusers import DiffusionPipeline
import gradio as gr
import torch


class ModelText2image(c.Module):
    """
    To use Text2Image the users can make image from the prompt.
    """
    def __init__(self, config = None, basemodel: str = 'Linaqruf/anything-v3.0', **kwargs):
        self.set_config(config, kwargs=kwargs)
        self.basemodel = basemodel
        self.pipeline = DiffusionPipeline.from_pretrained(self.basemodel, torch_dtype=torch.float16)        
        self.pipeline.to("cuda")

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config.sup)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    

    def generate(self, basemodel='Linaqruf/anything-v3.0', Prompt='a cat sitting on the chair', SaveFileName="./newimage"):
        """
        Creates image from a given prompt.

        Optional parameters:

        Prompt:           Prompt to generate image.
        SaveFileName:     FileName to save the results.

        """
        self.basemodel = basemodel
        self.pipeline = DiffusionPipeline.from_pretrained(self.basemodel, torch_dtype=torch.float16)        
        self.pipeline.to("cuda")

        image=self.pipeline(Prompt).images[0]
        image.save(f"{SaveFileName}.jpg")
        c.print(f"New image created. {SaveFileName}.jpg")
        return image


    def gradio(self):
        """
        This is function for gradio
        """
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    model_input = gr.Textbox(
                        value='Linaqruf/anything-v3.0', 
                        label='Model Select https://huggingface.co/models?other=diffusers:StableDiffusionPipeline&sort=trending', 
                        placeholder='Type in the base model name-hubname/modelname',
                        interactive=True,
                    )
                    prompt_input = gr.Textbox(
                        value='A cat swimming in the red river',
                        label='Prompt',
                        placeholder='Prompting...',
                        interactive=True,
                    )
                    savefilename_input = gr.Textbox(
                        value='cat',
                        label='Saving Path',
                        interactive=True,
                    )
                    submit_btn = gr.Button('Submit')
                image_box = gr.Image(
                    interactive=False,
                    visible=True,
                )
            
            
            submit_btn.click(self.generate, inputs=[model_input, prompt_input, savefilename_input], outputs=[image_box])

            
        demo.launch(share=True)