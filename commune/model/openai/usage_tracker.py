
import commune as c
from transformers import AutoTokenizer

class UsageTracker(c.Module):
    def __init__(self,
                  tokenizer='gpt2',
                  max_output_tokens=10_000_000,
                  max_input_tokens=10_000_000, 
                 **kwargs):

        self.max_output_tokens = max_output_tokens
        self.max_input_tokens = max_input_tokens

        self.set_tokenizer(tokenizer)
        self.usage = {
            'input': 0,
            'output': 0,
        }
        self.start_time = c.time()

    
    @property
    def age(self):
        return c.time() - self.start_time
    
    def usage_check(self ):
        too_many_output_tokens = self.usage['output'] < self.max_output_tokens
        too_many_input_tokens = self.usage['input'] < self.max_input_tokens
        return bool(too_many_output_tokens and too_many_input_tokens)
    

    def register_tokens(self, prompt:str, mode='input'):
        if not isinstance(prompt, str):
            prompt = str(prompt)
        input_tokens = self.num_tokens(prompt)
        self.usage[mode] += input_tokens

        assert self.usage_check(), \
                f"Too many tokens,output: {self.max_input_tokens} {self.max_output_tokens} output tokens, {self.usage}"
    
        return {'msg': f"Registered {input_tokens} {mode} tokens", 'success': True}
    


    def set_tokenizer(self, tokenizer: str = 'gpt2'):

        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast= True)
        except ValueError:
            print('resorting ot use_fast = False')
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token    
        self.tokenizer = tokenizer

        return self.tokenizer

    def num_tokens(self, text:str) -> int:
        num_tokens = 0
        tokens = self.tokenizer.encode(text)

        if len(tokens) == 0:
            return 0
        if isinstance(tokens, list) and isinstance(tokens[0], list):
            for i, token in enumerate(tokens):
                num_tokens += len(token)
        else:
            num_tokens = len(tokens)
        return num_tokens


