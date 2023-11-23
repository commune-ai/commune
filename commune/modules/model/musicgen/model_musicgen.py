import commune as c
from tempfile import NamedTemporaryFile
import torch

import gradio as gr
import os
from audiocraft.models import MusicGen

from audiocraft.data.audio import audio_write

MODEL = None

class ModelMusicgen(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)

    
    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config.sup)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y

    def load_model(self, version):
        print("Loading model", version)
        return MusicGen.get_pretrained(version)
    
    
    def predict(self, model, text, melody, duration, topk, topp, temperature, cfg_coef):
        """This is the predict function that will generate aduio

        Args:
            model : "melody"
            text : user prompt
            melody : model
            duration : audio duration
            topk : topk
            topp : topp
            temperature : temperature
            cfg_coef : cfg_coef
            
        Returns:
            audio
        """
        global MODEL
        topk = int(topk)
        if MODEL is None or MODEL.name != model:
            MODEL = self.load_model(model)
    
        if duration > MODEL.lm.cfg.dataset.segment_duration:
            raise gr.Error("MusicGen currently supports durations of up to 30 seconds!")
        MODEL.set_generation_params(
            use_sampling=True,
            top_k=topk,
            top_p=topp,
            temperature=temperature,
            cfg_coef=cfg_coef,
            duration=duration,
        )
    
        if melody:
            sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t().unsqueeze(0)
            print(melody.shape)
            if melody.dim() == 2:
                melody = melody[None]
            melody = melody[..., :int(sr * MODEL.lm.cfg.dataset.segment_duration)]
            output = MODEL.generate_with_chroma(
                descriptions=[text],
                melody_wavs=melody,
                melody_sample_rate=sr,
                progress=False
            )
        else:
            output = MODEL.generate(descriptions=[text], progress=False)
    
        output = output.detach().cpu().float()[0]
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(file.name, output, MODEL.sample_rate, strategy="loudness", add_suffix=False)
            waveform_video = gr.make_waveform(file.name)
        return waveform_video


    
    def gradio(self):
        """
        This is the gradio function that will generate UI for user.
        """
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        text = gr.Text(label="Input Text", interactive=True)
                        melody = gr.Audio(source="upload", type="numpy", label="Melody Condition (optional)", interactive=True)
                    with gr.Row():
                        submit = gr.Button("Submit")
                    with gr.Row():
                        model = gr.Radio(["melody"], label="Model", value="melody", interactive=True)
                        duration = gr.Slider(minimum=1, maximum=30, value=10, label="Duration", interactive=True)
                    with gr.Row():
                        topk = gr.Number(label="Top-k", value=250, interactive=True)
                        topp = gr.Number(label="Top-p", value=0, interactive=True)
                        temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                        cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
                with gr.Column():
                    with gr.Row():
                        output = gr.Video(label="Generated Music")
            submit.click(self.predict, inputs=[model, text, melody, duration, topk, topp, temperature, cfg_coef], outputs=[output])
        
        demo.launch(share=True, quiet=True)