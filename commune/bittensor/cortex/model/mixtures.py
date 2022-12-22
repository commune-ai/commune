from typing import Sequence

import bittensor
import torch
from torch import nn
from torch import tensor as T

from cortex.models.embedding import SentenceEmbedder


class BasicMixtureModel(nn.Module):
    def __init__(
        self, model_names: Sequence[str], enc1_dim: int = 128, num_queried: int = 5
    ):

        super().__init__()

        self.num_queried = num_queried
        self.sentence_embedder = SentenceEmbedder()

        self.encoders_1 = nn.ModuleDict()
        for model in model_names:
            self.encoders_1[model] = nn.Linear(bittensor.__network_dim__, enc1_dim)
        self.act = torch.nn.ReLU()

        cat_encode_dim = enc1_dim * num_queried
        self.encoder_2 = nn.Linear(cat_encode_dim, cat_encode_dim)

        self.decoder = nn.Linear(cat_encode_dim, bittensor.__vocab_size__, bias=False)

    def forward(self, hidden_states: Sequence[T], models: Sequence[str]) -> T:

        assert len(hidden_states) == len(models) == self.num_queried

        enc = []
        for h, key in zip(hidden_states, models):
            if h.dtype != torch.float32:
                h = h.to(torch.float32)
            enc.append(self.encoders_1[key](h))

        enc = torch.cat(enc, axis=-1)
        enc = self.act(enc)

        enc = self.encoder_2(enc)
        enc = self.act(enc)

        logits = self.decoder(enc)

        return logits
