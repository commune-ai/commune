from langchain import PromptTemplate
from typing import Union, Dict, Callable, List
from langchain.chains import TransformChain
import commune

class TransformBlock(TransformChain, commune.Block):
    
    def __init__(self, **kwargs):
        '''
        Args:
            kwargs:
                input_variables: List[str]  
                output_variables: List[str]
                transfrom: Callable,

        '''
    
        TransformChain.__init__(self, **kwargs)  
        
        
    
    @classmethod
    def default(cls):  
        
        def default_transform_func(inputs: dict) -> dict:
            text = inputs["text"]
            shortened_text = "\n\n".join(text.split("\n\n")[:3])
            return {"output_text": [shortened_text]}
    
        return cls(input_variables=["text"], output_variables=["output_text"], transform=default_transform_func)


commune.module(TransformBlock)

if __name__ == '__main__':
    block  = TransformBlock.default()
    print(block('What is the meaning of life?'))
