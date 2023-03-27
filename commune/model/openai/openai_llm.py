
import openai
# import commune
import os

# class OpenAILLM(commune.Moudle):
class OpenAILLM:
    def __init__(self,
                 model: str = "text-davinci-003",
                temperature: float=0.9,
                max_tokens: int=10,
                top_p: float=1.0,
                frequency_penalty: float=0.0,
                presence_penalty: float=0.0,
                prompt: str = None,
                key: str = 'OPENAI_API_KEY'
                ):
        self.set_key(key)
        self.set_params(locals())
        
    def set_params(self, params: dict):
        assert isinstance(params, dict), "params must be a dict."
        param_keys = ['model', 'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']
        self.params = {}
        for key in param_keys:
            self.params[key] = params[key]
            
        
        
        
        
    def set_key(self, key: str = None) -> None:
        openai.api_key = os.getenv(key, None)
        
        if isinstance(key, str) and openai.api_key is None:
            openai.api_key = key
        assert openai.api_key is not None, "OpenAI API key not found."
            
        
    # @classmethod
    # def install_env(cls):
    #     cls.cmd("pip install openai")

    def forward(self,prompt="Who is Alan Turing? what is the point of life?",
            params: dict = None,
            text_only: bool = True
    ) -> str:
        
        params = params if params != None else self.params

        print(f"Running OpenAI LLM with params: {params}", 'blue')

        
        print(f"Running OpenAI LLM with prompt: {prompt}", 'yellow')

        response = openai.Completion.create(
            prompt=prompt, 
            **params
        )
        text = response['choices'][0]['text']
        commune.print('Result: '+text, 'green')

        if text_only:
            return text
        return response

    @classmethod
    def test(cls, key: str = 'OPENAI_API_KEY'):
        
        model = OpenAILLM(key=key)
        print(model.run(prompt="Who is Alan Turing?"))        

if __name__ == "__main__":
    OpenAILLM.test(key='INSERT YOUR KEY HERE')


