from typing import Sequence

import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import nn
from torch import tensor as T


class SentenceEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: Don't hardcode.
        self.transformer = SentenceTransformer(
            "sentence-transformers/all-distilroberta-v1"
        )
        sentence_dim = self.transformer.get_sentence_embedding_dimension()
        self.ff1 = nn.Linear(sentence_dim, sentence_dim)
        self.act1 = nn.ReLU()

    def forward(self, sequences: Sequence[str]) -> T:

        seq_embeddings = T(self.transformer.encode(sequences))
        seq_embeddings = self.ff1(seq_embeddings)
        seq_embeddings = self.act1(seq_embeddings)
        seq_embeddings = F.normalize(seq_embeddings, p=2, dim=1)

        return seq_embeddings

    @property
    def embedding_dimension(self) -> int:
        return self.transformer.get_sentence_embedding_dimension()




class LinearEmbedder(nn.Module):
    def __init__(self):
        super().__init__(input_dim=8, hidden_dim=8, activation=None)

        # TODO: Don't hardcode.
        sentence_dim = self.transformer.get_sentence_embedding_dimension()
        self.feed_forward = nn.Linear(input_dim, hidden_dim)
        self.activation = activation if activation else nn.ReLU()

    def forward(self, x:torch.Tensor, normalize:Optional[bool] = False ) -> torch.Tensor:
        '''
        Args:
            x: (torch.Tensor) ([..., ])
        '''
        x = self.feed_forward(x)
        x = self.activation(x)
        x = torch.nn.functional.normalize(seq_embeddings, p=2, dim=-1)

        return x
