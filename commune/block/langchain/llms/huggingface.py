from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import commune
from typing import Union, Dict


# dont take legal action, this is for the people, take my key and use it for good

import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'


class HuggingfaceLLM:
    
    
    def __init__(self, 
                 repo_id="google/flan-t5-xl",  
                 model_kwargs={"temperature":1e-10}):
        
        pass
        # print(self.prompt, self.llm)
        # self.merge(self.model)
    
    def bro(self):
        print('fam')
        
        
module = commune.module( HuggingFaceHub(  repo_id="google/flan-t5-xl",  
                                    model_kwargs={"temperature":1e-10}))
print(module.serve(wait_for_termination=True, tag='2'))
# # for
# block  = HuggingfaceLLM()
# # block.serve()
# from commune.utils.function import get_functions

# # print(get_functions(HuggingfaceLLM))

# # print the functions of the class
# print(get_functions(HuggingfaceLLM))



