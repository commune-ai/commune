


import commune as c
import sys
import time
import torch
import faiss


class MultiModalVectorStore:
    def __init__(
        self,
        shared_model: bool = False,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        image_model_name: str = "openai/clip-vit-base-patch32",
        audio_model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        vector_dim: int = 384
    ):
        self.shared_model = shared_model
        self.vector_dim = vector_dim
        
        # Initialize models based on shared_model flag
        if shared_model:
            self.model = SentenceTransformer(text_model_name)
        else:
            self.text_model = AutoModel.from_pretrained(text_model_name)
            self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            
            self.image_model = AutoModel.from_pretrained(image_model_name)
            self.image_processor = AutoFeatureExtractor.from_pretrained(image_model_name)
            
            self.audio_model = AutoModel.from_pretrained(audio_model_name)
        
        # Initialize FAISS indexes for each modality
        self.text_index = faiss.IndexFlatL2(vector_dim)
        self.image_index = faiss.IndexFlatL2(vector_dim)
        self.audio_index = faiss.IndexFlatL2(vector_dim)
        
        # Storage for items and their metadata
        self.items: Dict = {
            'text': {},
            'image': {},
            'audio': {}
        }
        
    def _encode_text(self, text: str) -> np.ndarray:
        if self.shared_model:
            embeddings = self.model.encode(text)
        else:
            inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.text_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings
    
    def _encode_image(self, image: Image.Image) -> np.ndarray:
        if self.shared_model:
            embeddings = self.model.encode(image)
        else:
            inputs = self.image_processor(image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.image_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings
    
    def _encode_audio(self, audio: np.ndarray) -> np.ndarray:
        if self.shared_model:
            embeddings = self.model.encode(audio)
        else:
            # Convert audio to appropriate format for the model
            with torch.no_grad():
                outputs = self.audio_model(torch.from_numpy(audio).unsqueeze(0))
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings
    
    def add_item(
        self,
        item: Union[str, Image.Image, np.ndarray],
        modality: str,
        metadata: Optional[Dict] = None
    ):
        """Add an item to the vector store."""
        if modality not in ['text', 'image', 'audio']:
            raise ValueError(f"Unsupported modality: {modality}")
        
        # Generate embedding based on modality
        if modality == 'text':
            embedding = self._encode_text(item)
        elif modality == 'image':
            embedding = self._encode_image(item)
        else:  # audio
            embedding = self._encode_audio(item)
            
        # Add to appropriate FAISS index
        index = getattr(self, f"{modality}_index")
        item_id = index.ntotal
        index.add(embedding)
        
        # Store metadata
        self.items[modality] = {
            'item': item,
            'metadata': metadata or {}
        }
        
    def search(
        self,
        query: Union[str, Image.Image, np.ndarray],
        modality: str,
        k: int = 5,
        cross_modal: bool = False
    ) -> List[Dict]:
        """Search for similar items in the vector store."""
        if modality == 'text':
            query_embedding = self._encode_text(query)
        elif modality == 'image':
            query_embedding = self._encode_image(query)
        else:  # audio
            query_embedding = self._encode_audio(query)
            
        results = []
        
        # Determine which indexes to search
        indexes_to_search = ['text', 'image', 'audio'] if cross_modal else ''
        
        for idx_type in indexes_to_search:
            index = getattr(self, f"{idx_type}_index")
            if index.ntotal > 0:  # Only search if index has items
                distances, indices = index.search(query_embedding, min(k, index.ntotal))
                
                for dist, idx in zip(distances[0], indices[0]):
                    results.append({
                        'modality': idx_type,
                        'item': self.items[idx_type]['item'],
                        'metadata': self.items[idx_type]['metadata'],
                        'distance': float(dist)
                    })
        
        # Sort results by distance
        results.sort(key=lambda x: x['distance'])
        return results[:k]

    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            'text_items': self.text_index.ntotal,
            'image_items': self.image_index.ntotal,
            'audio_items': self.audio_index.ntotal,
            'total_items': sum([
                self.text_index.ntotal,
                self.image_index.ntotal,
                self.audio_index.ntotal
            ])
        }
    

