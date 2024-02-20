import http.client
import commune as c
from openai import OpenAI
import gradio as gr
from io import BytesIO
import base64
from PIL import Image

class DallE(c.Module):
    
    whitelist = ['generate', 'edit', 'variation', 'gradio']

    def __init__(self, api_key:str = None, host='api.openai.com/v1/images', cache_key:bool = True):
        config = self.set_config(kwargs=locals())
        self.conn = http.client.HTTPSConnection(self.config.host)
        self.set_api_key(api_key=config.api_key, cache=config.cache_key)
        
    def set_api_key(self, api_key:str, cache:bool = True):
        if api_key == None:
            api_key = self.get_api_key()

        self.api_key = api_key
        if cache:
            self.add_api_key(api_key)

        assert isinstance(api_key, str)

    def generate( self, 
                prompt: str, # required; max len is 1000 for dall-e-2 and 4000 for dall-e-3
                model: str = "dall-e-2", # "dall-e-2" | "dall-e-3"
                n: int = 1, # number of images Must be between 1 and 10. For dall-e-3, only n=1
                quality: str = "standard",  # "standard" | "hd" only supported for dall-e-3
                response_format: str = "url", # "url" or "b64_json"
                size: str = "1024x1024", #  256x256, 512x512, 1024x1024 for dall-e-2.  1024x1024, 1792x1024, 1024x1792 for dall-e-3
                style: str = "vivid", # "vivid" | "natural", only supported for dall-e-3
                api_key: str = None, # api_key "sk-..."
                ) -> str: 
        api_key = api_key if api_key != None else self.api_key

        # Create a client object with api key
        client = OpenAI(api_key = api_key)

        # Get generated response 
        response = client.images.generate(
            prompt=prompt,
            model=model,
            n=n,
            quality=quality,
            response_format=response_format,
            size=size,
            style=style,
        )

        return response
    
    def edit( self, 
            prompt: str, # required; max len is 1000 for dall-e-2
            image: str, # required; PNG file, less than 4MB,
            mask: str, # fully transparent areas; PNG file, less than 4MB,
            model: str = "dall-e-2", # "dall-e-2"
            n: int = 1, # number of images Must be between 1 and 10
            response_format: str = "url", # "url" or "b64_json"
            size: str = "1024x1024", #  256x256, 512x512, 1024x1024 
            api_key: str = None, # api_key "sk-..."
        ) -> str: 
        api_key = api_key if api_key != None else self.api_key

        client = OpenAI(api_key = api_key)

        # Get edited response 
        response = client.images.edit(
            prompt=prompt,
            model=model,
            image=open(image, "rb"),
            mask=open(mask, "rb"),
            n=n,
            response_format=response_format,
            size=size,
        )

        return response
    
    def variation( self, 
                image: str, # required; PNG file, less than 4MB,
                model: str = "dall-e-2", # "dall-e-2"
                n: int = 1, # number of images Must be between 1 and 10
                response_format: str = "url", # "url" or "b64_json"
                size: str = "1024x1024", #  256x256, 512x512, 1024x1024 
                api_key: str = None, # api_key "sk-..."
                ) -> str: 
        api_key = api_key if api_key != None else self.api_key

        client = OpenAI(api_key = api_key)

        # Get variation response 
        response = client.images.create_variation(
            model=model,
            image=open(image, "rb"),
            n=n,
            response_format=response_format,
            size=size,
        )

        return response
        
    def generateFromGradio(self, prompt):
        response = self.generate(prompt=prompt, response_format='b64_json', size="256x256")

        # Get Image object from the base64 encoded data
        return Image.open(BytesIO(base64.b64decode(response.data[0].b64_json)))
    
    def editFromGradio(self, prompt, image):
        response = self.edit(prompt = prompt, image = image, response_format='b64_json', size="256x256")
        
        return Image.open(BytesIO(base64.b64decode(response.data[0].b64_json)))

    
    def variationFromGradio(self, image):
        response = self.variation(image = image, response_format='b64_json', size="256x256")

        return Image.open(BytesIO(base64.b64decode(response.data[0].b64_json)))

    def gradio(self):
        with gr.Blocks() as demo:
            with gr.Tab("Generate Testing"):
                with gr.Row():
                    # Left column (inputs); prompt, generate button
                    with gr.Column():
                        input_gen=gr.Text(label="prompt") 
                        button_gen = gr.Button("Generate")
                    # Right column (outputs); generated image
                    with gr.Column():
                        output_gen = gr.Image(label="generated image")            
                    # Bind function to output_gen
                    button_gen.click(fn=self.generateFromGradio, inputs=input_gen, outputs=output_gen)

            with gr.Tab("Edit Testing"):
                with gr.Row():
                    # inputs: {prompt, image, mask}, edit button
                    with gr.Column():
                        input_edit=[gr.Text(label="prompt") , gr.Image(label="image", type="filepath"), gr.Image(label="mask", type="filepath")]
                        button_edit = gr.Button("Edit")
                    # edited image
                    with gr.Column():
                        output_edit = gr.Image(label="edited image")
                    # Bind function to button_edit
                    button_edit.click(fn=self.editFromGradio, inputs=input_edit, outputs=output_edit)

            with gr.Tab("Variation Testing"):
                with gr.Row():
                    # image, change button
                    with gr.Column():
                        input_var = gr.Image(label="image", type="filepath")
                        button_var = gr.Button("Change")
                    # changed image
                    with gr.Column():
                        output_var = gr.Image(label="changed image")            
                    # Bind function to button_var
                    button_var.click(fn=self.variationFromGradio, inputs=input_var, outputs=output_var)
        # Launch the gradio block and share
        demo.launch(quiet=True, share=True)
