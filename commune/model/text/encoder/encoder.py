from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict
import commune as c


class TextEncoder(c.Module):
    def __init__(self,
                 model: str = 'gpt125m', 
                 tokenizer = None,):
        self.shortcuts = c.module('model.transformer').shortcuts
        self.set_tokenizer(tokenizer if tokenizer else model)
        self.set_model(model)
        
    def set_tokenizer(self, tokenizer:str) ->   None:

        tokenizer = self.shortcuts.get(tokenizer, tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    
    def set_model(self, model:str) ->  None:
        model = self.shortcuts.get(model, model)

        self.model = AutoModel.from_pretrained(model)
    
    @staticmethod
    def mean_pooling(model_output, attention_mask: torch.Tensor)-> torch.Tensor:
        #Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @classmethod
    def test(cls, 
             sentences: List[str] = ['This framework generates embeddings for each input sentence',
                'Sentences are passed as a list of string.',
                'The quick brown fox jumps over the lazy dog.'],
             **kwargs
             ):
        self = cls(**kwargs)
        embeddings = self.encode(sentences)
        
        return embeddings
    

    def encode(self, sentences: List[str]) -> List[List[float]]:
        #Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

        #Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        #Perform pooling. In this case, mean pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])