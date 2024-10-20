import torch

def decode_topk(  forward_response_tensor: torch.Tensor, topk:int=4096, vocab_size:int=50257) -> torch.Tensor:
    """ Returns full logits by decoding topk-encoding input. """
    batch_size, sequence_len, _ = forward_response_tensor.shape
    encoded_probs = forward_response_tensor  # encoded probabilities: [batch_size, sequence_len, topk + topk]
    topk_values = encoded_probs[..., :topk]  # topk probs: [batch_size, sequence_len, topk]
    topk_indices = encoded_probs[..., topk:].long()  # topk probs indices: [batch_size, sequence_len, topk]

    topk_pmass = topk_values.sum(dim=-1)  # topk probability mass: [batch_size, sequence_len]
    remainder_pmass = torch.clamp(1 - topk_pmass, 1e-40, 1)  # remainder probability mass: [batch_size, sequence_len]
    
    st.write(remainder_pmass, vocab_size, topk, 'bro')
    remainder_floor = remainder_pmass / (vocab_size - topk)  # divide remainder: [batch_size, sequence_len]

    logits = torch.ones((batch_size, sequence_len, vocab_size), dtype=topk_values.dtype).to(topk_values.device)
    logits *= torch.log(remainder_floor)[:, :, None]  # set probability floor: [batch_size, sequence_len, vocab_size]

    logits.scatter_(-1, topk_indices, torch.log(topk_values + 1e-40))  # insert topk probs: [batch_size, sequence_len, vocab_size]

    return logits  # [batch_size, sequence_len, vocab_size]

def decode_topk(  forward_response_tensor: torch.Tensor, topk:int=4096, vocab_size:int=None) -> torch.Tensor:
    """ Returns full logits by decoding topk-encoding input. """
    batch_size, sequence_len, _ = forward_response_tensor.shape
    encoded_probs = forward_response_tensor  # encoded probabilities: [batch_size, sequence_len, topk + topk]
    topk_values = encoded_probs[..., :topk]  # topk probs: [batch_size, sequence_len, topk]
    topk_indices = encoded_probs[..., topk:].long()  # topk probs indices: [batch_size, sequence_len, topk]

    topk_pmass = topk_values.sum(dim=-1)  # topk probability mass: [batch_size, sequence_len]
    remainder_pmass = torch.clamp(1 - topk_pmass, 1e-40, 1)  # remainder probability mass: [batch_size, sequence_len]
    remainder_floor = remainder_pmass / (vocab_size - topk)  # divide remainder: [batch_size, sequence_len]

    logits = torch.ones((batch_size, sequence_len, vocab_size), dtype=topk_values.dtype).to(topk_values.device)
    logits *= torch.log(remainder_floor)[:, :, None]  # set probability floor: [batch_size, sequence_len, vocab_size]

    logits.scatter_(-1, topk_indices, torch.log(topk_values + 1e-40))  # insert topk probs: [batch_size, sequence_len, vocab_size]

    return logits  # [batch_size, sequence_len, vocab_size]
import torch

def encode_topk( forward_response_tensor: torch.Tensor , topk:int=4096) -> torch.Tensor:
   
    """ Returns topk tokens/probabilities given unnormalized logits as input. """

    #import ipdb; ipdb.set_trace()

    logits = forward_response_tensor  # unnormalized logit scores: [batch_size, sequence_len, vocab_size]
    probs = torch.softmax(logits, dim=-1).to(torch.float32)  # normalized probabilities: [batch_size, sequence_len, vocab_size]

    topk_indices = torch.argsort(probs, dim=-1, descending=True)[...,:topk]
    # topk_values, topk_indices = torch.topk(probs, topk) # topk probs and indices: [batch_size, sequence_len, topk]

    topk_values = probs.gather( index=topk_indices, dim=-1)
    encoded_probs = torch.cat([topk_values, topk_indices], dim=-1)  # [batch_size, sequence_len, topk + topk]
    return encoded_probs  # [batch_size, sequence_len, topk + topk]


def decode_topk(  forward_response_tensor: torch.Tensor, topk:int=4096, vocab_size:int=None) -> torch.Tensor:
    """ Returns full logits by decoding topk-encoding input. """
    batch_size, sequence_len, _ = forward_response_tensor.shape
    encoded_probs = forward_response_tensor  # encoded probabilities: [batch_size, sequence_len, topk + topk]
    topk_values = encoded_probs[..., :topk]  # topk probs: [batch_size, sequence_len, topk]
    topk_indices = encoded_probs[..., topk:].long()  # topk probs indices: [batch_size, sequence_len, topk]

    topk_pmass = topk_values.sum(dim=-1)  # topk probability mass: [batch_size, sequence_len]
    remainder_pmass = torch.clamp(1 - topk_pmass, 1e-40, 1)  # remainder probability mass: [batch_size, sequence_len]
    remainder_floor = remainder_pmass / (vocab_size - topk)  # divide remainder: [batch_size, sequence_len]

    logits = torch.ones((batch_size, sequence_len, vocab_size), dtype=topk_values.dtype).to(topk_values.device)
    logits *= torch.log(remainder_floor)[:, :, None]  # set probability floor: [batch_size, sequence_len, vocab_size]

    logits.scatter_(-1, topk_indices, torch.log(topk_values + 1e-40))  # insert topk probs: [batch_size, sequence_len, vocab_size]

    return logits  # [batch_size, sequence_len, vocab_size]
