import commune as c
import PIL.Image
import gradio as gr
import base64
import time
import os
import google.generativeai as genai
from dotenv import load_dotenv


class Gemini(c.Module):
    def __init__(self):        
        # Set Google API key 
        load_dotenv()
        # Set Google API key
        google_token = os.getenv("GOOGLE_API_KEY")
        print(f"-----------------{google_token}--------------------")
        genai.configure(api_key = google_token)
        # Create the Model
        self.txt_model = genai.GenerativeModel('gemini-pro')
        self.vis_model = genai.GenerativeModel('gemini-pro-vision')

    # Image to Base 64 Converter
    def image_to_base64(self, image_path):
        with open(image_path, 'rb') as img:
            encoded_string = base64.b64encode(img.read())
        return encoded_string.decode('utf-8')

    # Function that takes User Inputs and displays it on ChatUI
    def query_message(self, history, txt, img):
        if not img:
            history += [(txt, None)]
            return history
        base64_img = self.image_to_base64(img)
        data_url = f"data:image/jpeg;base64,{base64_img}"
        history += [(f"{txt} ![]({data_url})", None)]
        return history

    # Function that takes User Inputs, generates Response and displays on Chat UI
    def llm_response(self, history, text, img):
        if not img:
            response = self.txt_model.generate_content(text)
            history += [(None, response.text)]
            return history

        else:
            img = PIL.Image.open(img)
            response = self.vis_model.generate_content([text, img])
            history += [(None, response.text)]
            return history

    def gradio(self):
        # Interface Code
        with gr.Blocks() as demo:
            with gr.Row():
                image_box = gr.Image(type="filepath")
            
                chatbot = gr.Chatbot(
                    scale = 2,
                    height=750
                )
            text_box = gr.Textbox(
                    placeholder="Enter text and press enter, or upload an image",
                    container=False,
                )

            btn = gr.Button("Submit")
            
            # Add functions to components and actions
            
            clicked = btn.click(self.query_message,
                                [chatbot, text_box, image_box],
                                chatbot
                                ).then(self.llm_response,
                                        [chatbot, text_box, image_box],
                                        chatbot
                                        )
            
        # demo.queue()
        demo.launch(share=True)