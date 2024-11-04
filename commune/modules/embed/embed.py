from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity
import numpy as np
from typing import List, Union

class TextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the TextEmbedder with a pre-trained model.
        Args:
            model_name (str): Name of the pre-trained model to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling of token embeddings
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Get embeddings for input texts
        Args:
            texts: Single text or list of texts to embed
        Returns:
            torch.Tensor: Embeddings for the input texts
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = 
            
        # Tokenize texts
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        # Perform pooling
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def calculate_similarity(self, text1: Union[str, List[str]], text2: Union[str, List]) -> torch.Tensor:
        """
        Calculate semantic similarity between two texts or lists of texts
        Args:
            text1: First text or list of texts
            text2: Second text or list of texts
        Returns:
            torch.Tensor: Similarity scores
        """
        embeddings1 = self.get_embeddings(text1)
        embeddings2 = self.get_embeddings(text2)
        
        return cosine_similarity(embeddings1, embeddings2)

    def test()
        embedder = TextEmbedder()
        
        # Example texts
        text1 = "I love programming"
        text2 = "I enjoy coding"
        text3 = "The weather is nice today"
        
        # Calculate similarities
        similarity = embedder.calculate_similarity(text1, text2)
        print(f"Similarity between '{text1}' and '{text2}': {similarity.item():.4f}")
        
        # Multiple texts
        texts1 = [text1, text2]
        texts2 = [text2, text3]
        similarities = embedder.calculate_similarity(texts1, texts2)
        print("\nSimilarity matrix:")
        print(similarities)