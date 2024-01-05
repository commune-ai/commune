import http.client
import json
import commune as c
import base64

from PIL import Image
from io import BytesIO



class Corcel(c.Module):
    

    def __init__(self, api_key:str = None, host='api.corcel.io', cache_key:bool = True):
        if api_key == None:
            api_key = c.choice(self.api_keys())
        config = self.set_config(kwargs=locals())
        self.api_key = config.api_key
        c.print(config)
        self.conn = http.client.HTTPSConnection(self.config.host)
    

    def image(self,
              content= "Give me an image of a cute, phantom, cockapoo. Very cute, not too fluffy",
              miners_to_query:int = 1,
              top_k_miners_to_query:int = 40,
              ensure_responses:bool = True,
              miner_uids = [],
              messages = [],
              model = "cortext-image",
              style = "vivid",
              size = "1024x1024",
              quality = "hd",): 

        request = {
        "miners_to_query": miners_to_query,
        "top_k_miners_to_query": top_k_miners_to_query,
        "ensure_responses": ensure_responses,
        "miner_uids": miner_uids,
        "messages": content,
        "model": model,
        "style": style,
        "size": size,
        "quality": quality
        }

        payload = json.dumps(request)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': self.api_key
        }

        self.conn.request("POST", "/cortext/image", payload, headers)

        res = self.conn.getresponse()
        return res.read().decode("utf-8")

    
    
    def generate(self, 
                content:str,
                history:list=None,
                stream:bool = False,
                miners_to_query:int = 1,
                top_k_miners_to_query:int = 40,
                ensure_responses:bool = True,
                miner_uids = [],
                model = "cortext-ultra",
                role = "user",
                ) -> str:
        api_key =  self.api_key
        
        # build payload

        payload =  {
            "miners_to_query": miners_to_query,
            "top_k_miners_to_query": top_k_miners_to_query,
            "ensure_responses": ensure_responses,
            "miner_uids": miner_uids,
            "messages": [

                {
                "role": role,
                "content": content
                }
            ],
            "model": model,
            "stream": stream
            }

        if history is not None:
            assert isinstance(history, list)
            payload["messages"] =  [history + payload["messages"]]

        c.print(payload)
        # make API call to BitAPAI
        payload = json.dumps(payload)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': api_key
        }        
        self.conn.request("POST", "/cortext/text", payload, headers)

        # fetch responses
        res = self.conn.getresponse()
        data = res.read().decode("utf-8")

        data = json.loads(data)

        # find non-empty responses in data['choices']
        return data[0]['choices'][0]['delta']['content']
    

    def imagine( self, 
                prompt: str,
                negative_prompt: str = '',
                width: int = 1024,
                height: int = 1024,
                count:int = 20,
                return_all:bool = False,
                exclude_unavailable:bool = True,
                uids: list = None,
                api_key: str = None, 
                history: list=None) -> str: 
        api_key = api_key if api_key != None else self.api_key
        
        # build payload
        payload =  {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "count": count,
            "return_all": return_all,
            "exclude_unavailable": exclude_unavailable,
            "width": width,
            "height": height
        }
        if uids is not None:
            payload['uids'] = uids

        if history is not None:
            assert isinstance(history, list)
            assert len(history) > 0
            assert all([isinstance(i, dict) for i in history])
            payload = payload[:1] +  history + payload[1:]


        # make API call to BitAPAI
        payload = json.dumps(payload)
        headers = {
            'Content-Type': 'application/json',
            'X-API-KEY': api_key
        }
        self.conn.request("POST", "/image", payload, headers)


        # fetch responses
        res = self.conn.getresponse()
        data = res.read().decode("utf-8")
        data = json.loads(data)
        
        assert len(data['choices']) > 0
        
        # find non-empty responses in data['choices']
        for i in range(len(data['choices'])):
            if len(data['choices'][i]['images']) > 0:

                # [display image using gradio]
                # import gradio as gr
                # with gr.Blocks() as demo:
                #     gallery = gr.Gallery()
                #     gallery.value = data['choices'][i]['images']
                #     # im = gr.Image(label="Image Generated by Commune BitAPAI module")
                #     # im.value = Image.open(BytesIO(base64.b64decode(data['choices'][i]['images'][0][23:])))  # exclude preceeding 'data:image/jpeg;base64,'
                # demo.launch(share=True)

                # [display image using streamlit]
                # import streamlit as st
                # st.image(
                #     image=data['choices'][i]['images'][0],
                #     caption="Image Generated by Commune BitAPAI module"
                # )
                
                # [display image using PIL]
                image_data = data['choices'][i]['images'][0].split(',')[1]  # remove the 'data:image/jpeg;base64,' prefix (assuming the base64 encoded image is stored in a variable called 'image_data')
                image_data = bytes(image_data, encoding='utf-8')  # convert to bytes
                img = Image.open(BytesIO(base64.b64decode(image_data)))  # decode and open the image using PIL
                img.show()  # display the image using PIL

                return data['choices'][i]['images'][0]

        return None

