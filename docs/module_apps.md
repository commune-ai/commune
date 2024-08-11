In order to make an app, please ensure you have streamlit installed. We will support other frameworks in the future, like gradio, and html based apps. 

To make a streamlit app, make sure to include an def app function in your module. This will be the function that will be called when the app is run.

c new_module agi

Copy this code in it and save it as agi.py

```python

import commune as c
import streamlit as st

class AGI(c.Module):
    def __init__(self):
        self.model = c.module('model.openai')()

    def app(self):
        st.title("Hello AGI friends")
        st.write("This is a simple app to show how to make a streamlit app with commune")
        text_input = st.text_input("Enter your name", "Type Here")
        submit_button = st.button("Submit")
        if submit_button:
            output = self.model.generate(input_text)
            st.write(output)

if __name__ == "__main__":
    AGI.run()

```
App.run()
```

If you want to run the app as a standalone app, you can run the following command in the terminal

```bash
c app agi port=8501
```
or
```bash
streamlit run chatapp.py --server.port 8501
```
or 



