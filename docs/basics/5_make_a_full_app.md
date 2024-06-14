In order to make an app, please ensure you have streamlit installed. We will support other frameworks in the future, like gradio, and html based apps. 

To make a streamlit app, make sure to include an def app function in your module. This will be the function that will be called when the app is run.

c new_module chatapp

enter the code here

```python

import commune as c


class ChatApp(c.Module):
    def __init__(self):
        super().__init__()

    def app(self):
        st.title("Hello chat city")

# =
App.run(__name__)

```

If you want to run the app as a standalone app, you can run the following command in the terminal

```bash
streamlit run chatapp.py --server.port 8501
```
