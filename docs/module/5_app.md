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

# This is necessary for streamlit apps.
App.run(__name__)

```

To run the app, you can use the following command:

```bash
c app chatapp
```
