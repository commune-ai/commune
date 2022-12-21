from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio 


import os, sys
sys.path.append(os.getenv('PWD'))
asyncio.set_event_loop(asyncio.new_event_loop())

import commune
import streamlit as st



class GatewayServer(commune.Module):
    def __init__(self):
        commune.Module.__init__(self)

    def __call__(self, data:dict, metadata:dict={}):
        input = data['input']
        output_data = data
        return {'data': output_data, 'metadata': metadata}


    def call_function(self, module:str, fn:str, args:list= [], kwargs:dict={} , launch_kwargs:dict={}, launch_args:list=[]):
        module =  self.launch(module=module, *launch_args, **launch_kwargs )
        output = getattr(module, fn)(*args, **kwargs)
        return output



model = commune.launch(module=GatewayServer, actor=False)
st.write(model.call_function(module='commune.dataset.text.huggingface', fn='sample'))
st.write(model.list_modules())


# server = commune.server.ServerModule(module = model ,ip='0.0.0.0',  port=8902)
# server.start()
# client = commune.server.ClientModule(ip=server.ip, port=server.port)
# st.write(client.forward(data={'input': 'hey'}))
# st.write(server)
# data = model.forward(['hey man, how is it', 'I have a red car and it sells for'])
# st.write(data)
# st.write(model.tokenizer.batch_decode(data))