
import commune
import numpy as np
from commune.gradio.typing import Image, Block, Textbox, Label
import gradio as gr



class ExampleGradioSchemaBuilder(commune.Module):

    def __init__(self):
        ...



    @staticmethod
    def image_classifier(inp : Image) -> Label:
        return {'cat': 0.3, 'dog': 0.7}

    @staticmethod
    def message_received(msg : Textbox) -> Textbox:
        return msg


    @classmethod
    def interface(cls) -> Block:
        def add_text(history, text):
            history = history + [(text, None)]
            return history, gr.update(value="", interactive=False)

        def add_file(history, file):
            history = history + [((file.name,), None)]
            return history

        def bot(history):
            response = "**That's cool!**"
            history[-1][1] = response
            return history

        with gr.Blocks() as demo:
            chatbot = gr.Chatbot([], elem_id="chatbot").style(height=500)

            with gr.Row():
                with gr.Column(scale=0.85):
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press enter, or upload an image",
                    ).style(container=False)
                with gr.Column(scale=0.15, min_width=0):
                    btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])

            txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
                bot, chatbot, chatbot
            )
            txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
            file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
                bot, chatbot, chatbot
            )

        return demo