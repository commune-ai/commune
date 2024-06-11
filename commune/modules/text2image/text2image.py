import commune as c

class Text2Image(c.Module):
    @classmethod
    def dashboard(cls):
        import streamlit as st
        self = cls()
        module = c.connect('stability')
        image_path = st.text_input('image_path', 'image.png')
        image = module.text_to_image('hello world')
        

Text2Image.run(__name__)