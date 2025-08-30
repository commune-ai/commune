import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import re

class SentenceDistanceCalculator:
    def __init__(self, model_name="bert-base-uncased", use_stopwords=True):
        """
        Initialize the SentenceDistanceCalculator with PyTorch and HuggingFace models.
        
        Args:
            model_name (str): HuggingFace model to use for embeddings
            use_stopwords (bool): Whether to remove stopwords during preprocessing
        """
        self.use_stopwords = use_stopwords
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Common English stopwords
        if use_stopwords:
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
                'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
                'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
                'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
                'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
                't', 'can', 'will', 'just', 'don', 'should', 'now'
            }
    
    def preprocess(self, sentence):
        """
        Preprocess a sentence by tokenizing and optionally removing stopwords.
        
        Args:
            sentence (str): Input sentence
            
        Returns:
            list: Processed tokens
        """
        # Simple tokenization using regex
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        
        if self.use_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
            
        return tokens
    
    def get_embedding(self, sentence):
        """
        Get embedding for a sentence using the transformer model.
        
        Args:
            sentence (str): Input sentence
            
        Returns:
            torch.Tensor: Sentence embedding
        """
        # Tokenize and prepare input
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use mean pooling over all tokens
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_maskcc
        return embedding.squeeze()
    
    def jaccard_distance(self, sentence1, sentence2):
        """
        Calculate Jaccard distance between two sentences.
        
        Args:
            sentence1 (str): First sentence
            sentence2 (str): Second sentence
            
        Returns:
            float: Jaccard distance (0-1, where 0 means identical)
        """
        tokens1 = set(self.preprocess(sentence1))
        tokens2 = set(self.preprocess(sentence2))
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        if union == 0:
            return 0.0
        
        return 1.0 - (intersection / union)

    def levenshtein_distance(self, sentence1, sentence2):
        """
        Calculate normalized Levenshtein (edit) distance between two sentences with PyTorch.
        
        Args:
            sentence1 (str): First sentence
            sentence2 (str): Second sentence
            
        Returns:
            float: Normalized Levenshtein distance (0-1, where 0 means identical)
        """
        tokens1 = self.preprocess(sentence1)
        tokens2 = self.preprocess(sentence2)
        
        # Create a matrix
        rows = len(tokens1) + 1
        cols = len(tokens2) + 1
        distance = torch.zeros((rows, cols), dtype=torch.int)
        
        # Initialize the matrix
        for i in range(rows):
            distance[i, 0] = i
        for j in range(cols):
            distance[0, j] = j
            
        # Fill the matrix
        for i in range(1, rows):
            for j in range(1, cols):
                if tokens1[i-1] == tokens2:
                    cost = 0
                else:
                    cost = 1
                distance[i, j] = min(
                    distance[i-1, j] + 1,      # deletion
                    distance[i, j-1] + 1,      # insertion
                    distance[i-1, j-1] + cost  # substitution
                )
        
        # Normalize the distance
        max_len = max(len(tokens1), len(tokens2))
        if max_len == 0:
            return 0.0
        
        return distance[-1, -1].item() / max_len
    
    def transformer_distance(self, sentence1, sentence2):
        """
        Calculate distance between two sentences using transformer embeddings.
        
        Args:
            sentence1 (str): First sentence
            sentence2 (str): Second sentence
            
        Returns:
            float: Cosine distance (0-1, where 0 means identical)
        """
        embedding1 = self.get_embedding(sentence1)
        embedding2 = self.get_embedding(sentence2)
        
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
        
        # Convert to distance
        return 1.0 - cos_sim.item()

    def test(self):
        """
        Test function to demonstrate the SentenceDistanceCalculator class.
        """
        # Test pairs of sentences
        sentence_pairs = [
            ("The quick brown fox jumps over the lazy dog.", 
            "The fast brown fox leaps over the sleepy dog."),
            
            ("Python is a programming language.", 
            "Java is a programming language."),
            
            ("I love to eat pizza.", 
            "Pizza is my favorite food."),
            
            ("The weather is nice today.", 
            "The weather is terrible today."),
            
            ("This is a completely different sentence.", 
            "No similarity whatsoever between these two.")
        ]
        
        print("Sentence Distance Comparison:\n")
        
        for i, (s1, s2) in enumerate(sentence_pairs):
            print(f"Pair {i+1}:")
            print(f"Sentence 1: {s1}")
            print(f"Sentence 2: {s2}")
            
            jaccard = self.jaccard_distance(s1, s2)
            levenshtein = self.levenshtein_distance(s1, s2)
            transformer = self.transformer_distance(s1, s2)
            
            print(f"Jaccard Distance: {jaccard:.4f}")
            print(f"Normalized Levenshtein Distance: {levenshtein:.4f}")
            print(f"Transformer Embedding Distance: {transformer:.4f}")
            # print("-" * 50)