import commune as c
import gradio as gr
from transformers import pipeline
from PIL import Image
import numpy as np
import tensorflow as tf
import mediapy
import os
import sys
from huggingface_hub import snapshot_download
from image_tools.sizes import resize_and_crop

os.system("git clone https://github.com/google-research/frame-interpolation")
sys.path.append("frame-interpolation")
from eval import interpolator, util

ffmpeg_path = util.get_ffmpeg_path()
mediapy.set_ffmpeg(ffmpeg_path)

model = snapshot_download(repo_id="akhaliq/frame-interpolation-film-style")
interpolator = interpolator.Interpolator(model, None)

class ModelImage2video(c.Module):
    """
    This is the class that convert image to video model.
    """
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config.sup)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y

    def resize(self, width, img):
        """
        This is the function that resize the image same as specific width.

            Paramaters:
                width (int): basic width
                img (Image): image to resize

            Return:
                Nothing
        """
        basewidth = width
        img = Image.open(img)
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        return img

    def resize_img(self, img1, img2, output_name):
        """
        This is the function that resize the image according to the specific image.

            Parameters:
                img1 (str): target image
                img2 (str): image to resize
                output_name(str): name of the output

            Return:
                Nothing
        """
        img_target_size = Image.open(img1)
        img_to_resize = resize_and_crop(
            img2,
            (img_target_size.size[0], img_target_size.size[1]),
            crop_origin="middle"
        )
        img_to_resize.save(output_name)

    def generate_interpolation(self, frames, times_to_interpolate, fps):
        """
        This is the main generation function.

            Parameters:
                frames([]): instances of the image files
                times_to_interpolate(int): It's the number of times to interpolate
                fps(int): It's the number of fps for the output video

            Return:
                It returns one video file.
        """
        resized_frames = []
        for i, frame in enumerate(frames):
            resized_frame = self.resize(256, frame.name)
            resized_frame.save(f"test{i+1}.png")
            resized_frames.append(f"test{i+1}.png")

        for i in range(1, len(resized_frames)):
            self.resize_img(resized_frames[0], resized_frames[i], f"resized_img{i+1}.png")

        input_frames = [resized_frames[0]]
        for i in range(1, len(resized_frames)):
            input_frames.append(f"resized_img{i+1}.png")

        frames = list(util.interpolate_recursively_from_files(input_frames, times_to_interpolate, interpolator))

        mediapy.write_video("out.mp4", frames, fps=fps)
        
        return "out.mp4"

    def gradio(self):
        """
        This function is for gradio UI generation.
            Parameters:
                Nothing
            Return: 
                Nothing
        Gradio provides for users to do all things in the UI
        """
        demo = gr.Blocks()

        with demo:
            with gr.Row():
            
                # Left column (inputs)
                with gr.Column():

                    with gr.Row():
                        # upload images and get image strings
                        input_arr = [
                            gr.inputs.File(type="file", label="Images", file_count="multiple")
                        ]

                    with gr.Row():
                        input_arr.append(gr.inputs.Slider(minimum=2, maximum=10, step=1, label="Times to Interpolate"))
                        input_arr.append(gr.inputs.Slider(minimum=15, maximum=60, step=1, label="fps"))
                                
                    # Rows of instructions & buttons
                    with gr.Row():
                        button_gen_video = gr.Button("Generate Video")

                        
                # Right column (outputs)
                with gr.Column():
                    output_interpolation = gr.Video(label="Generated Video")
                    
            # Bind functions to buttons
            button_gen_video.click(fn=self.generate_interpolation, inputs=input_arr, outputs=output_interpolation)

        demo.launch(quiet=True, enable_queue=True, share=True)
