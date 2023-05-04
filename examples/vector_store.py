import commune
import torch

# Define a new class that inherits from commune.Module
config = dict(
    
)
class VectorStore(commune.Module):
    def __init__(self, config = None):
        self.set_config(config)
        self.vectors = vectors
    
    def set_model(self, model):
        assert hasattr(model, 'encode')
        self.model = model
        
    def encode(self, text:str, **kwargs):
        return self.model.embed(text, **kwargs)
        
    def search(self, query):
        query_vector = torch.tensor(query)
        similarities = torch.einsum('ij,kj->i', query_vector, self.vectors)
        indices = similarities.argsort(descending=True)
        return indices.tolist()

# Create some random vectors to use as our vector store

# Launch the vector store as a public server
VectorStore.launch(name='vector_store', kwargs={'vectors': vectors})

# Connect to the vector store and perform a vector search
vector_store = commune.connect('vector_store')
query = [0.1, 0.2, 0.3, ..., 0.9]
results = vector_store.search(query)
print(results)
