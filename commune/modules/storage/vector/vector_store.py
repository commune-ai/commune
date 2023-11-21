import commune as c
import torch

class VectorStore(c.Module):
    def __init__(self, 
                    config = None,
                    **kwargs
                 ):
        self.set_config(config=config, kwargs=kwargs)
        self.k2index = {}
        self.index2k = {}
        self.vectors = []
        self.example_vector = None
        config = self.set_config(config)
        self.set_model(config.model)
        
        

    def set_model(self, model='model'):
        self.model = c.connect(model)
        
        
        
    def encode(self, text:str, **kwargs):
        return self.model.encode(text, **kwargs)
    

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
    
    @classmethod
    def resolve_model(self, model=None):
        if model == None:
            model = self.model
        elif isinstance(model, str):
            model = c.connect(model)
            
        return model
        

    def embed(self, text:str, model=None, **kwargs):
        model = self.resolve_model(model)
        vectors = self.model.embed(text, **kwargs)
    
    def add_vector(self, k, v , verbose=True):
    

        v = self.resolve_vector(v)
        stacked_vectors = []
        if len(self.vectors) == 0:
            stacked_vectors = [v]
        else:
            stacked_vectors = [self.vectors, v]
        
        self.vectors = torch.stack(stacked_vectors)
        idx = len(self.vectors) - 1
        if verbose:
            c.print(f'Adding vector {k} at index {idx}, {self.vectors}')
        self.k2index[k] = idx
        self.index2k[idx] = k
        
    def rm_vector(self, k):
        last_idx = len(self.vectors) - 1
        print(f'last_idx: {last_idx}')
        # swap the vector to be removed with the last vector
        idx = self.k2index[k]
        last_k = self.index2k[last_idx]
        last_v = self.vectors[last_idx]
        self.vectors[idx] = last_v
        self.k2index[last_k] = idx
        self.index2k[idx] = last_k
        # remove the last vector
        del self.index2k[last_idx]
        del self.k2index[last_k]
        
        self.vectors = self.vectors[:-1]




    def search(self, query, top_k=10, chunks=1):
        assert len(self.vectors) > 0, 'No vectors stored in the vector store'
        c.print(f'query: {query}', 'vector shape:', self.vectors.shape)
        query_vector = torch.tensor(query) 
        if len(query_vector.shape) == 1:
            query_vector = query_vector.unsqueeze(0)
        # Convert the query into a tensor (assuming it's a vector representation)
        print(query_vector.shape, self.vectors.shape)
        similarities = torch.einsum('ij,kj->i', query_vector, self.vectors)  # Compute the similarity scores between the query and stored vectors
        indices = similarities.argsort(descending=True)[:top_k]  # Sort the similarity scores and retrieve the indices of the top-k results
        results = {}  # Create an empty dictionary to store the results
        indices = indices.tolist()  # Convert the indices tensor to a list
        for idx in indices:
            key = self.index2k[idx]  # Retrieve the key corresponding to the index
            score = similarities[idx].item()  # Retrieve the similarity score for the corresponding index
            results[key] = score  # Store the key-score pair in the results dictionary
        return results  # Return the dictionary of results


    @classmethod
    def test(cls):
        self = cls()
        self.add_vector('test', [1,2,3])
        assert self.search([1,2,3]) == {'test': 14.0}
        self.rm_vector('test')
        assert len(self.vectors) == 0
        print('test passed')
        
        
        
