import bittensor
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import commune

class BittensorLLM(LLM):

    topk = 10
    def __init__(self, 
                 **kwargs):
        LLM.__init__(self,**kwargs)
    
        
    @property
    def _llm_type(self) -> str:
        return "bittensor"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[:self.topk]
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"topk": self.topk}


if __name__ == "__main__":
    llm =  BittensorLLM()
    print(llm("What is the meaning of life?"))
