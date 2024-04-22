import commune as c
import replicate
import http.client
import os
from moviepy.editor import VideoFileClip
from PIL import Image
import gradio as gr

class Video2Text(c.Module):  
  """
  Video2Text is a class that allows users to get description from the short video
  """

  def __init__(self, api_key:str = None, host='replicate.com', cache_key:bool = True):
      config = self.set_config(kwargs=locals())
      self.conn = http.client.HTTPSConnection(self.config.host)
      self.set_api_key(api_key=config.api_key, cache=config.cache_key)

      self.image2text_models = {
        "gfodor/instructblip":               "gfodor/instructblip:ca869b56b2a3b1cdf591c353deb3fa1a94b9c35fde477ef6ca1d248af56f9c84",
        "j-min/clip-caption-reward":         "j-min/clip-caption-reward:de37751f75135f7ebbe62548e27d6740d5155dfefdf6447db35c9865253d7e06",
        "pharmapsychotic/clip-interrogator": "pharmapsychotic/clip-interrogator:8151e1c9f47e696fa316146a2e35812ccf79cfc9eba05b11c7f450155102af70",
        "rmokady/clip_prefix_caption":       "rmokady/clip_prefix_caption:9a34a6339872a03f45236f114321fb51fc7aa8269d38ae0ce5334969981e4cd8",
        }

      self.summarize_models = {
        "replicate/flan-t5-small": "replicate/flan-t5-small:69716ad8c34274043bf4a135b7315c7c569ec931d8f23d6826e249e1c142a264",
        "daanelson/flan-t5-large": "daanelson/flan-t5-large:ce962b3f6792a57074a601d3979db5839697add2e4e02696b3ced4c022d4767f",
        "replicate/flan-t5-xl":    "replicate/flan-t5-xl:7a216605843d87f5426a10d2cc6940485a232336ed04d655ef86b91e020e9210",
        "nateraw/samsum-llama-7b": "nateraw/samsum-llama-7b:16665c1f00ad4d5d6c88393aa05390cf4d9f4e49c8abde4c58f2e1e71fd806f9",
      }

  def set_api_key(self, api_key:str, cache:bool = True):
      if api_key == None:
          api_key = self.get_api_key()

      self.api_key = api_key
      if cache:
          self.add_api_key(api_key)

      # Set the REPLICATE API TOKEN
      os.environ["REPLICATE_API_TOKEN"] = api_key
      assert isinstance(api_key, str)

  def extract_images(self, video, interval):
    video_clip = VideoFileClip(video)

    folder = './frames'
    # Create the directory
    os.makedirs(folder, exist_ok=True)

    # Clear the directory
    for filename in os.listdir(folder):
      file_path = os.path.join(folder, filename)
      try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
          os.unlink(file_path)
      except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Extract frames of the video
    for i in range(0, int(video_clip.duration), interval):
      frame = video_clip.get_frame(i)
      new_img = Image.fromarray(frame)
      new_img.save(f'./frames/frame{i//interval}.png')

    video_clip.close()

  def describe(self, 
               api_key : str = None, # api key
               image2text_model: str = 'gfodor/instructblip', # image2text model
               video: str = "test_video.mp4", # video file path
               interval: int = 5, # time interval
               summarize: bool = True, # summarize the desc
               summarize_model: str = 'replicate/flan-t5-xl', # summarize model
               summarize_max_len: int = 50, # Maximum number of tokens to generate. A word is generally 2-3 tokens
               summarize_top_p: float = 0.95, # When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens
               summarize_temperature : float = 0.7 # Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.
               ):    
    if api_key != None:
      os.environ["REPLICATE_API_TOKEN"] = api_key

    # Extract frames of the video
    self.extract_images(video=video, interval=interval)
    descriptions = []

    # Get desc for each frame
    for file in os.listdir("./frames"):

      input = {}
      if image2text_model == 'gfodor/instructblip':
        input={
          "image_path": open("frames/" + file, "rb"),
          "prompt": "Describe the scene of this frame from a video",
        }
      else:
        input={
          "image": open("frames/" + file, "rb")
        }

      output = replicate.run(
        self.image2text_models[image2text_model],
        input
      )

      c.print(file, output)
      descriptions.append(output)

    # Return only desc
    if summarize == False:
      return {"desc": descriptions}      

    # Summarize the description of all frames
    output = replicate.run(
      self.summarize_models[summarize_model],
      input={
        "prompt": "Summarize the description of this video from the descriptions of its frames." + "\n".join(descriptions),
        "max_length": summarize_max_len,
        "top_p": summarize_top_p,
        "temperature": summarize_temperature,
      }
    )
    full_response = ""
    for item in output:
      full_response += item

    c.print("Summarized: ", full_response)

    return "\n".join(descriptions), full_response


  def gradio(self):
    with gr.Blocks() as demo:
      with gr.Row():
        with gr.Column():             
          api_key = gr.Text(label = "api_key")
          video = gr.Video(label = "video", interactive = True)
          button = gr.Button("Describe")
        with gr.Column():
          description = gr.Textbox(lines = 3, label = "description")
          summary = gr.Text(label = "summary")
        button.click(fn = self.describe, inputs = [api_key, video], outputs = [description, summary])

    demo.launch(quiet=True, share=True)