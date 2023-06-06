import commune as c
import torch
import numpy as np
from typing import List, Union, Literal
from safetensors import safe_open
from safetensors.torch import save_file


class VectorStore(c.Module):
    """A class for storing vectors and searching them using vector similarity metrics"""
    def __init__(self, 
                    config = None
                 ):
        self.k2index = {}
        self.index2k = {}
        self.vectors = []
        self.example_vector = None
        config = self.set_config(config)
        self.set_model(config.model)
        
        

    def set_model(self, model='model'):
        self.model = c.connect(model)
        
        
    def deploy_model(cls, model, *args, **kwargs):
        return c.deploy()
        
    def encode(self, text:str, **kwargs):
        """Encodes a text into a vector representation"""
        return self.model.embed(text, **kwargs)
    

    def resolve_vector(self, v:torch.Tensor):
        """Converts a vector to a torch.Tensor and checks that it has the same shape as the example vector"""
        if isinstance(v, list):
            v = torch.tensor(v)
        elif isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
            
        assert isinstance(v, torch.Tensor), f'Expected a torch.Tensor, got {type(v)}'
        if len(v.shape) == 1:
            v = v.unsqueeze(0)
        return v
    
    @classmethod
    def resolve_model(self, model=None):
        """Converts a model to a torch.Tensor and checks that it has the same shape as the example vector"""
        if model == None:
            model = self.model
        elif isinstance(model, str):
            model = c.connect(model)
            
        return model
        

    def embed(self, text:str, model=None, **kwargs):
        model = self.resolve_model(model)
        return self.model.embed(text, **kwargs)
    
    def append(self, k, v, verbose=True):
    
        v = self.resolve_vector(v)
        stacked_vectors = []
        if len(self.vectors) == 0:
            stacked_vectors = [v]
        else:
            stacked_vectors = [self.vectors, v]
            
        
        self.vectors = torch.cat(stacked_vectors, dim=0)
        idx = len(self.vectors) - 1
        if verbose:
            c.print(f'Adding vector {k} at index {idx}, {self.vectors}')
        self.k2index[k] = idx
        self.index2k[idx] = k
        
    def pop(self, k):
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

        similarities = torch.einsum('ij,kj->i', query_vector, self.vectors)  # Compute the similarity scores between the query and stored vectors
        indices = similarities.argsort(descending=True)[:top_k]  # Sort the similarity scores and retrieve the indices of the top-k results
        results = {}  # Create an empty dictionary to store the results
        indices = indices.tolist()  # Convert the indices tensor to a list
        for idx in indices:
            key = self.index2k[idx]  # Retrieve the key corresponding to the index
            score = similarities[idx].item()  # Retrieve the similarity score for the corresponding index
            results[key] = score  # Store the key-score pair in the results dictionary
        return results  # Return the dictionary of results

    def cosine_similarity(self, v1 : Union[list, np.ndarray, torch.Tensor], v2 : Union[list, np.ndarray, torch.Tensor]):
        return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))

    @classmethod
    def manhattan_distance(cls, v1 : Union[list, np.ndarray, torch.Tensor], v2 : Union[list, np.ndarray, torch.Tensor]):
        return torch.sum(torch.abs(v1 - v2))

    @classmethod
    def save(cls, name : str='memory', method : Literal["safetensors", "pickle", "torch"]='safetensors'):
        self = cls()
        dir = f"{self.dirpath()}/"
        
        if method == 'safetensors':
            file = f'{name}.safetensors'
            dir += file
            mem : dict = {f"{name}" : torch.ones((5,100, 100))}
            save_file(mem, dir); self.putc(name , file)

        elif method == 'pickle':
            file = f'{name}.pickle'
            # save_file(self.vectors, file)
        

    @classmethod
    def load(cls, key : str="memory" , method : Literal["safetensors", "pickle", "torch"]='safetensors', *args, **kwargs):
        # FIX ME DOES NOT WORK
        self = cls()
        path = f"{self.dirpath()}/{self.getc(key)}"
        tensors = {}
        if method == 'safetensors':
            with safe_open(path, *args, **kwargs) as f:
                for keys in f.keys():
                    tensors[keys] = f.get_tensor(keys)
        return tensors

    def __repr__(self):
        return f'VectorStore with {len(self.vectors)} vectors'