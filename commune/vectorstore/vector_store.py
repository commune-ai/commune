import commune
import torch

# Define a new class that inherits from commune.Module
config = dict(
    
)
class VectorStore(commune.Module):
    def __init__(self, config = None):
        self.k2index = {}
        self.index2k = {}
        self.vectors = []
        self.example_vector = None
        self.set_config(config)

    def set_model(self, model):
        assert hasattr(model, 'encode')
        self.model = model
        
    def encode(self, text:str, **kwargs):
        return self.model.embed(text, **kwargs)
    

    def resolve_vector(self, v:torch.Tensor):
        
        if isinstance(v, list):
            v = torch.tensor(v)
        elif isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
            
        assert isinstance(v, torch.Tensor), f'Expected a torch.Tensor, got {type(v)}'
        if self.example_vector == None:
            self.example_vector = v
        assert self.example_vector.shape == v.shape, f'Expected a vector of shape {self.example_vector.shape}, got {v.shape}'
        
        return v
    

    def add_vector(self, k, v ):
    

        v = self.resolve_vector(v)
        self.vectors.append(v)
        idx = len(self.vectors) - 1
        self.k2index[k] = idx
        self.index2k[idx] = k
        
    def rm_vector(self, k):
        last_idx = len(self.vectors) - 1
        # swap the vector to be removed with the last vector
        idx = self.k2index[k]
        last_k = self.index2k[last_idx]
        last_v = self.vectors[last_idx]
        self.vectors[idx] = last_v
        self.k2index[last_k] = idx
        self.index2k[idx] = last_k
        # remove the last vector
        del self.index2k[last_idx]
        del self.vectors[last_idx]
        del self.k2index[k]

        
    def search(self, query, top_k=10):
        query_vector = torch.tensor(query)
        similarities = torch.einsum('ij,kj->i', query_vector, self.vectors)
        indices = similarities.argsort(descending=True)[:top_k]
        results = {}
        for idx in indices:
            key = self.index2k[idx]
            score = similarities[idx].item()
            results[key] = score
        return results


    @classmethod
    def test(cls):
        self = cls()
        self.add_vector('test', [1,2,3])
        assert self.search([1,2,3]) == {'test': 14.0}
        self.rm_vector('test')
        assert self.search([1,2,3]) == {}
        print('test passed')
        
        
        
