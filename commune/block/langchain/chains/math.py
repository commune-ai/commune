from langchain.chains import LLMMathChain
from langchain import HuggingFaceHub
import os

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'


class MathChain(LLMMathChain):
    
    def __init__(self, llm, **kwargs):
        LLMMathChain.__init__(self, llm=llm, **kwargs)
        
    @classmethod
    def default(cls):

        llm = HuggingFaceHub(repo_id="google/flan-t5-xl",  model_kwargs={"temperature":1e-10}, huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'])
        return cls(llm=llm)
    
    



model = MathChain.default()
print(model.run('2+2+5'))
