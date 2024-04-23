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
        module = self.config.model.module
        model_kwargs = model

        self.model = commune.module(module)(**model_kwargs)
        
        
    def forward(self, **kwargs):
      return self.model(**kwargs)
      
      
    @classmethod
    def sandbox(cls):
      self = cls()
      
      
if __name__ == "__name__":
  KnowledgeGraph.sandbox()
  
  
