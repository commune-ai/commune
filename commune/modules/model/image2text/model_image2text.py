import commune as c
import torch 
import re 
import gradio as gr
from PIL import Image

from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel 
import os
import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class ModelImage2text(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)
        self.device='cpu'

        self.model_id = "nttdataspain/vit-gpt2-stablediffusion2-lora"
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_id)

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config.sup)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y

    def predict(self, image):
        img = image.convert('RGB')
        self.model.eval()
        pixel_values = self.feature_extractor(images=[img], return_tensors="pt").pixel_values
        with torch.no_grad():
            output_ids = self.model.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True).sequences

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds[0]

    def newgradio(self):
        input = gr.inputs.Image(label="Upload any Image", type = 'pil', optional=True)
        output = gr.outputs.Textbox(type="text",label="Captions")
        with gr.Blocks() as demo:
            with gr.Row():
                    with gr.Column(scale=1):
                        img = gr.inputs.Image(label="Upload any Image", type = 'pil', optional=True)
                        button = gr.Button(value="Convert")
                    with gr.Column(scale=1):
                        out = gr.outputs.Textbox(type="text",label="Details")   
                        
            button.click(self.predict, inputs=[img], outputs=[out])
        
        demo.launch(share=True, quiet=True)