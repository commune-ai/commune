from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio 


import os, sys
sys.path.append(os.getenv('PWD'))
asyncio.set_event_loop(asyncio.new_event_loop())

import commune
import streamlit as st



class RemoteModel:
    def __init__(self, model="EleutherAI/gpt-neo-125M"):

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        if  self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    def getattr(self, k):
        return getattr(self, k)


    def forward(self, input:str="This is the first sentence. This is the second sentence.", tokenize=False):
    
        input = self.tokenizer(
                input, add_special_tokens=False, return_tensors="pt", padding=True,
            ).to(self.model.device)
        st.write(input.input_ids)
        return self.model(**input)

    def __call__(self, data:dict, metadata:dict={}):
        input = data['input']
        logits = self.forward(input = input).logits
        output_data = {'logits': logits}
        return {'data': output_data, 'metadata': metadata}



if __name__ == "__main__":

    model = commune.launch(module=RemoteModel, actor=False)

    st.write(model)
    server = commune.server.ServerModule(module = model ,ip='0.0.0.0')
    st.write(server.port )
    server.start()
    st.write(server.module)
    client = commune.server.ClientModule(ip=server.ip, port=server.port)

    st.write(client.forward(data={'input': 'hey, whadup'}))


    # st.write(server)
    # data = model.forward(['hey man, how is it', 'I have a red car and it sells for'])
    # st.write(data)
    # st.write(model.tokenizer.batch_decode(data))