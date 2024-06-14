import commune as c
import os
import gradio as gr
from scipy.io.wavfile import write
import subprocess

def inference(audio):
  os.makedirs("out", exist_ok=True)
  write('test.wav', audio[0], audio[1])
  os.system("python3 -m demucs.separate -n htdemucs --two-stems=vocals -d cpu test.wav -o out")
  return "./out/htdemucs/test/vocals.wav","./out/htdemucs/test/no_vocals.wav"
    
title = "Separate your music file"
description = "Drag and drop an audio file to easily separate it!"
article = ""

class Demusics(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config.sup)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y

    def split(self, input_file="test.mp3", bitrate=192):
        command = [
            "python3",
            "-m",
            "demucs",
            "--mp3",
            f"--mp3-bitrate {bitrate}",
            input_file
        ]

        try:
            subprocess.run(command, check=True)
            print("Splitting completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error while splitting: {e}")

    
    def gradio(self):
        examples=[[]]
        gr.Interface(
            inference, 
            gr.Audio(type="numpy", label="Song"),
            [gr.Audio(type="filepath", label="Vocals"),gr.Audio(type="filepath", label="Instrumentals")],
            title=title,
            description=description,
            article=article,
            examples=examples
            ).launch(enable_queue=True)
