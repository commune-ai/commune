import commune
st = commune.get_streamlit()




import commune



# llm = OpenAI(temperature=0, model_name="text-davinci-002")
# react = initialize_agent(tools, llm, agent="react-docstore", verbose=True)

from typing import *


import commune
class Judge(commune.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_tokenizer(self.config['tokenizer'])
        self.set_llm(self.config['llm'])
    def set_llm(self, llm = None):
        from langchain import OpenAI
        try:
            import openai
        except ModuleNotFoundError:
            self.run_command('pip install openai')
            import openai
        self.llm = self.launch(**llm)
    def set_tokenizer(self, tokenizer: str):

        if tokenizer == None and hasattr(self, 'tokenizer'):
            return self.tokenizer
                
        from transformers import AutoTokenizer

        if isinstance(tokenizer, str):
            self.config['tokenizer'] = tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast= True)
            except ValueError:
                print('resorting ot use_fast = False')
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        
        self.tokenizer = tokenizer
        
        
        return self.tokenizer
    def num_tokens(self, text:str, tokenizer: str = None, **kwargs) -> int:
        self.set_tokenizer(tokenizer)
        assert hasattr(self, 'tokenizer')
        return len(self.tokenizer(text,**kwargs)['input_ids'])
    @property
    def price_per_token(self):
        return self.config['price_per_token']
    def set_price_per_token(self, price_per_token:float = 0.0001) -> float:
        assert isinstance(price_per_token, float)
        self.config['price_per_token'] = price_per_token
        return price_per_token
    @classmethod
    def sandbox(cls):
        self = cls()
        st.write(self.num_tokens('bro whadup, how is it going fam whadup'))
        st.write(self.llm)




if __name__ == "__main__":
    
    Judge.run()





