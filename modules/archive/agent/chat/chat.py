import commune
st = commune.get_streamlit()




import commune



# llm = OpenAI(temperature=0, model_name="text-davinci-002")
# react = initialize_agent(tools, llm, agent="react-docstore", verbose=True)

from typing import *


import commune


class CustomChatbot(commune.Module):
    def __init__(self, llm:str= 'gpt4',
                 tokenizer:str='gpt2',
                 **kwargs):
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
    
    
    def chat(self, text:str) -> str:
        return 'Im not in the mood baby, leave me alone'   
    def num_tokens(self, text:str,
                   tokenizer: str = None, 
                   **kwargs) -> int:
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
    def test(cls):
        
        import commune
        chatbot =  commune.connect('chatbot')
        user_input = 'Hey Honey, how is it going?'
        ai_output = chatbot.chat(user_input)
        
        
        st.write(f'**USER INPUT**: {user_input}')
        st.write(f'**AI OUTPUT**: {ai_output}')
        
        # st.write(chatbot.chat('Hey Honey, how is it going?'))
        

        # del peer_info['intro']
        # del peer_info['examples']
        
        # st.write(peer_info)
        
        
        # st.multiselect('**My Modules**', live_servers, live_servers)
        
        # st.write(cls.servers())
        # st.write(self.num_tokens('bro whadup, how is it going fam whadup'))
        # st.write(self.llm.__dict__)

if __name__ == "__main__":
    
    CustomChatbot.run()






