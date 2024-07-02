import commune as c
import PIL.Image
import gradio as gr
import base64
import os
import google.generativeai as genai
from dotenv import load_dotenv


class Gemini(c.Module):
    def __init__(self):        
        # Set Google API key
        try:                
            load_dotenv()    
            # Set Google API key
            google_token = os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key = google_token)
        except IOError:
            print('Could not environment file. copy .env_copy to .env and add your own google api key.')        
            
        # Do configuration for gemini - temperature, top_k, top_p
        
        generation_config = {
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        # Create the Model
        self.txt_model = genai.GenerativeModel(model_name='gemini-pro',
                                               generation_config=generation_config,
                                               safety_settings=safety_settings
                                               )
                
        self.vis_model = genai.GenerativeModel(model_name='gemini-pro-vision',
                                               generation_config=generation_config,
                                               safety_settings=safety_settings
                                               )

    # Image to Base 64 Converter 
    # As our chat conversations include images, we need to show these Images in the chat.
    # We cannot just directly display images in chat as it is.
    # Hence one method is by providing the chat with an encoded base64 string.
    # Hence we will write a function that takes in the image path encodes it to base64 and returns a base64-encoded string.
    
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
            try:                
                img = PIL.Image.open(img)
            except IOError:
                print('Could not read image file.')
            response = self.vis_model.generate_content([text, img])
            history += [(None, response.text)]
            return history
    # Testing UI
    def gradio(self):
        # Interface Code
        with gr.Blocks() as demo:
            with gr.Row():
                image_box = gr.Image(type="filepath")
            
                chatbot = gr.Chatbot(
                    scale = 2,
                    height=750
                )
            
            with gr.Row():
                temperature_sld = gr.Slider(0, 2, value = 0, label="Temperature", info="Choose between 2 and 20", interactive=True)
                top_k_sld = gr.Slider(1, 100, value=1, step = 1, label="Top_k", info="Choose Top_k for gemini model", interactive=True)
                top_p_sld = gr.Slider(0, 1, value=0.9, label="Top_p", info="Choose Top_p for gemini model", interactive=True)
            text_box = gr.Textbox(
                    placeholder="Enter text and press enter, or upload an image",
                    container=False,
                )

            
            btn = gr.Button("Submit")
            
            # Add functions to components and actions
            def set_model(temperature_sld, top_k_sld, top_p_sld):                
                self.txt_model.temperature, self.txt_model.top_k, self.txt_model.top_p = temperature_sld, int(top_k_sld), top_p_sld
                self.vis_model.temperature, self.vis_model.top_k, self.vis_model.top_p = temperature_sld, top_k_sld, top_p_sld
                
            clicked = btn.click(self.query_message,
                                [chatbot, text_box, image_box],
                                chatbot
                                ).then(self.llm_response,
                                        [chatbot, text_box, image_box],
                                        chatbot
                                        )
            temperature_sld.change(set_model, inputs=[temperature_sld, top_k_sld, top_p_sld])
            top_k_sld.change(set_model, inputs=[temperature_sld, top_k_sld, top_p_sld])
            top_p_sld.change(set_model, inputs=[temperature_sld, top_k_sld, top_p_sld])
            
            
        # demo.queue()
        demo.launch(share=True)