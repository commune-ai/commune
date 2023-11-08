import commune as c
import torch
import shutil
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

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
        self.pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config.sup)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    
    def convert(self, prompt, steps=25, output="output"):
        """
        This is the converting function that convert prompt to video
        prompt: str
        steps: int = 25
        output: str = "output"
        """
        video_frames = self.pipe(prompt, num_inference_steps=steps).frames
        video_path = export_to_video(video_frames)
        shutil.copy(video_path, f"./{output}.mp4")
        c.print(f"Video file created successfully. {output}.mp4")