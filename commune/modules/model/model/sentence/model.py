
import commune as c
from sentence_transformers import SentenceTransformer

class Sentence(c.Module):
    def __init__(self, 
                model = 'sentence-transformers/all-MiniLM-L6-v2',
                device =  'cuda',
                  **kwargs):
        config = self.set_config(locals())
        self.set_model(model=config.model, device=config.device)
    
    def set_model(self, model:str, device:str):
        self.model = SentenceTransformer(model, device=device)

    def forward(self, text, **kwargs):
        initially_string = isinstance(text, str)
        if initially_string:
            text = [text]
        assert isinstance(text, list)
        assert isinstance(text[0], str)
        embeddings = self.model.encode(text, **kwargs)
        if initially_string:
            embeddings = embeddings[0]
        return embeddings
    def test(self):
        sentences = ["This is an example sentence", "Each sentence is converted"]
        embeddings = self.model.encode(sentences)
        c.print(embeddings.shape)
        sentences = "This is an example sentence"
        embeddings = self.model.encode(sentences)
        c.print(embeddings.shape)
        return embeddings
    
    
