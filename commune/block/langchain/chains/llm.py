from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import commune
from typing import Union, Dict


# dont take legal action, this is for the people, take my key and use it for good
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

class LLMChainBlock(LLMChain, commune.Block):
    
    def __init__(self, llm: 'LLM' = None, prompt: PromptTemplate = None ):
        self.set_chain(llm=llm, prompt=prompt)
        print(self.prompt, self.llm)
        
        
    def set_chain(self, llm: 'LLM' = None,  prompt: PromptTemplate = None, llm_chain: LLMChain = None):
        '''
        Set the CHAIN with the llm
        '''
        if isinstance(llm_chain, LLMChain):
            self.merge(llm_chain)
            
        prompt = prompt if prompt else self.default_prompt()
        llm = llm if llm else self.default_llm()
        LLMChain.__init__(self, prompt=prompt, llm=llm)
           
            
    @classmethod
    def default_prompt(cls) -> PromptTemplate:
        '''
        Default prompt template for the LLMChainBlock
        '''
        
        
        template = """Question: {question}
        Answer: Let's think step by step."""
        
        prompt = PromptTemplate(template=template, 
                                input_variables=cls.get_template_args(template))
        return prompt
        
    
    @classmethod
    def default_llm(cls) -> LLMChain:
        '''
        Default LLM for the LLMChainBlock
        
        '''
        
        
        return HuggingFaceHub(repo_id="google/flan-t5-xl",  model_kwargs={"temperature":1e-10}, huggingfacehub_api_token=)
    
        
    @classmethod
    def default(cls):
        llm = cls()
     
# for
block  = LLMChainBlock()
# block.serve()
print(block)
