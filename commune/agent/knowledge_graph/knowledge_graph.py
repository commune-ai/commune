import commune



prompt = """
Create the most important n relation tuples as follows (head, relation, tail) from the following text. Also, :
Ouput in a json format under {key} 

  {key}: List[Dict(head:str, relation:str, tail:str)]

}
"""




class KnowledgeGraph(commune.Module):
  
  
    prompt = prompt
     
     
    def set_prompt(self, prompt: str = None):

        if prompt is None:
            prompt = self.prompt
            
        self.prompt = prompt
  
  
    def set_model(self, model: 'Model'=  None):
        if model == None:
          model = dict(
                module = 'model.openai'
                model = "text-davinci-003",
                temperature=0.9,
                max_tokens=10,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                prompt = None,
                key = 'OPENAI_API_KEY'
          )
        module_path = model.pop('module')
        model_kwargs = model

        self.model = commune.get_module(module_path)(**model_kwargs)
        
        
    def forward(self, **kwargs):
      return self.model(**kwargs)
      
      
    @classmethod
    def sandbox(cls):
      self = cls()
      
      
if __name__ == "__name__":
  KnowledgeGraph.sandbox()
  
  
