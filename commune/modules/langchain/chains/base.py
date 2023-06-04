from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import commune
from typing import Union, Dict


# dont take legal action, this is for the people, take my key and use it for good

class BaseChain(LLMChain, commune.Block):
    
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
      
    def set_prompt(self, prompt: PromptTemplate):
        '''
        Set the prompt
        '''
        self.prompt = prompt
        
            
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
        
        
        return HuggingFaceHub(repo_id="google/flan-t5-xl",  model_kwargs={"temperature":1e-10},)
    
        
    @classmethod
    def default(cls):
        llm = cls()
     
# for
block  = BaseChain()
# block.serve()
print(block.run('Hello my name is billy'))
