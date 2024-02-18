import commune as c
import torch
import os
import random
import tempfile
import gradio as gr
import imageio
import numpy as np
import torch

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from .grstyle import getcss

DESCRIPTION = '# Text to Video Commune Module'
if (SPACE_ID := os.getenv('SPACE_ID')) is not None:
    DESCRIPTION += f'\nFor faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings.'

MAX_NUM_FRAMES = int(os.getenv('MAX_NUM_FRAMES', '200'))
DEFAULT_NUM_FRAMES = min(MAX_NUM_FRAMES,
                         int(os.getenv('DEFAULT_NUM_FRAMES', '16')))

class ModelText2video(c.Module):
    """
    ModelText2video is a class that allows us to make video from the user prompt
    Initialisation of this class is expensive because of the AI module,
    so we want initialize it once and then reuse the user's convert
    """

    def __init__(self, config = None, **kwargs):
        """
        Initialize the damo-vilab/text-to-video-ms-1.7b model.
        """
        self.set_config(config, kwargs=kwargs)
        self.pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b',
                                         torch_dtype=torch.float16,
                                         variant='fp16')
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config.sup)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y

    def to_video(self, frames: list[np.ndarray], fps: int) -> str:
        out_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir='./')
        print(out_file.name)
        writer = imageio.get_writer(out_file.name, format='FFMPEG', fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        return out_file.name


    def generate(self, prompt: str, seed: int, num_frames: int,
                 num_inference_steps: int) -> str:
        """
        This is the generating function that convert prompt to video
        prompt: str
        seed: int
        num_frames: int
        num_inference_steps: int
        """
        if seed == -1:
            seed = random.randint(0, 1000000)
        generator = torch.Generator().manual_seed(seed)
        frames = self.pipe(prompt,
                      num_inference_steps=num_inference_steps,
                      num_frames=num_frames,
                      generator=generator).frames
        return self.to_video(frames, 8)

    def newgradio(self):
        """
        This is the function for starting a new Gradio demo.
        """
        with gr.Blocks(css=getcss()) as demo:
            gr.Markdown(DESCRIPTION)
            with gr.Group():
                with gr.Box():
                    with gr.Row(elem_id='prompt-container').style(equal_height=True):
                        prompt = gr.Text(
                            label='Prompt',
                            show_label=False,
                            max_lines=1,
                            placeholder='Enter your prompt',
                            elem_id='prompt-text-input').style(container=False)
                        run_button = gr.Button('Generate video').style(
                            full_width=False)
                result = gr.Video(label='Result', show_label=False, elem_id='gallery')
                with gr.Accordion('Advanced options', open=False):
                    seed = gr.Slider(
                        label='Seed',
                        minimum=-1,
                        maximum=1000000,
                        step=1,
                        value=-1,
                        info='If set to -1, a different seed will be used each time.')
                    num_frames = gr.Slider(
                        label='Number of frames',
                        minimum=16,
                        maximum=MAX_NUM_FRAMES,
                        step=1,
                        value=16,
                        info=
                        'Note that the content of the video also changes when you change the number of frames.'
                    )
                    num_inference_steps = gr.Slider(label='Number of inference steps',
                                                    minimum=10,
                                                    maximum=50,
                                                    step=1,
                                                    value=25)
        
            inputs = [
                prompt,
                seed,
                num_frames,
                num_inference_steps,
            ]
        
            prompt.submit(fn=self.generate, inputs=inputs, outputs=result)
            run_button.click(fn=self.generate, inputs=inputs, outputs=result)
        
                
        demo.queue(api_open=False, max_size=15).launch(share=True, quiet=True)