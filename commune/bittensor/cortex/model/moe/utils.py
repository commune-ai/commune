import torch

def mix_response( response_success, normalized_topk_routing_scores):
    batch_size = response_success[0].shape[0]
    mixed_response = torch.zeros(batch_size, bittensor.__vocab_size__ + 1  , 2)
    all_logits = torch.tensor(list(range(bittensor.__vocab_size__)))
    mixed_response[:, : -1, 1] = all_logits.repeat(batch_size, 1)

    for r, w in list(zip(response_success, normalized_topk_routing_scores)):
        response = torch.zeros(batch_size, bittensor.__vocab_size__ , 2)
        response[:, :, 1] = all_logits.repeat(batch_size, 1)

        for batch in range(batch_size):
            r_batch = r[batch, : -1, :]
            r_batch_sorted = r_batch[r_batch[:, 0].sort(descending = False)[1]]
            index = r_batch_sorted[:, 1].long()
            prob = r_batch_sorted[:, 0] 
            response[batch, index, 0] = prob
        
        mixed_response[:, :-1, 0] += w * response[:, :, 0]

    for batch in range(batch_size):
        mixed_response[batch, -1, :] = torch.tensor([[0, -1]])

    return mixed_response

