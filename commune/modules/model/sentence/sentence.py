
import commune as c


class Sentence(c.Module):
    def __init__(self, 
                config=None,
                  **kwargs):
        config = self.set_config(config=config, kwargs=kwargs)
        return self.set_model(model=config.model, device=config.device)
    
    def set_model(self, model:str, device:str):
        c.ensure_package('sentence_transformers')
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, device=device)

    def forward(self, *sentences, **kwargs):
        embeddings = self.model.encode(sentences, **kwargs)
        return embeddings
    def test(self):
        sentences = ["This is an example sentence", "Each sentence is converted"]
        embeddings = self.model.encode(sentences)
        return embeddings
    print(embeddings)
