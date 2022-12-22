import torch
from typing import *

def phrase_cross_entropy(target_phrases: Union[List[List[int]], torch.Tensor],
                         topk_tensor: torch.Tensor,
                         ignore_index: int = -100, reduce=True, reduction='mean',
                         vocab_size_min: int = 50257) -> [torch.Tensor, torch.Tensor]:
    r"""
    Calculates the cross entropy of a phrase prediction against a target phrase, so that this is a multi-token
    extension of typical cross entropy calculated for next token prediction.
        Args:
            target_phrases (:obj:`List[List[int]]`, `required`):
                [batch_size, *] Target phrases in standard token sequence list.
            topk_tensor (:obj:`torch.Tensor`, `required`):
                [batch_size, (topk + 1), max_len] tensor includes topk token probabilities (prob_k) + floor_prob
                in first column with gradients attached, with std_tokens in remaining columns with ignore_index padding.
                Content structure:
                [[[prob_k=0_b=0, tok_0_k=0_b=0, tok_1_k=0_b=0, ..., ignore_index?],
                  [prob_k=1_b=0, tok_0_k=1_b=0, tok_1_k=1_b=0, ..., ignore_index?],
                  [...],
                  [prob_floor_b=0, ignore_index, ..., ignore_index]],
                 [[prob_k=0_b=1, tok_0_k=0_b=1, tok_1_k=0_b=1, ..., ignore_index?],
                  [prob_k=1_b=1, tok_0_k=1_b=1, tok_1_k=1_b=1, ..., ignore_index?],
                  [...],
                  [prob_floor_b=1, ignore_index, ..., ignore_index]],
                 [...]]
            ignore_index (:obj:`int`, `optional`):
                Padding value to use for unfilled token positions in a shorter token phrase.
            reduce (:obj:`bool`, `optional`):
                Whether to reduce the cross entropy over the batch dimension.
            reduction (:obj:`str`, `optional`):
                Reduction function to perform when reduce is True.
            vocab_size_min (:obj:`int`, `optional`):
                Minimum server vocab_size expected, should set to nominal 50257,
                used to prevent the floor_probs from being too large.
        Returns:
            loss_val (:obj:`torch.Tensor`, `required`):
                Validation cross entropy loss, either scalar if reduce or [batch_size].
            loss (:obj:`torch.Tensor`, `required`):
                Phrase cross entropy loss, either scalar if reduce or [batch_size].
    """

    batch_size, topk_p1, max_len = topk_tensor.shape  # [batch_size, (topk + 1), max_len]
    topk = topk_p1 - 1

    topk_tokens = topk_tensor[:, :-1, 1:].round().int()  # [batch_size, topk, max_len - 1] Phrase tokens with ignore_index token for padding.
    topk_probs = topk_tensor[:, :-1, 0]  # [batch_size, topk] Probabilities for each phrase in topk
    floor_probs = topk_tensor[:, -1, 0]  # [batch_size] Floor probabilities as mean probability for non-topk tokens

    topk_probs = torch.clamp(topk_probs, 0, 1)  # [batch_size, topk] ensure probabilities within [0, 1]
    floor_probs = torch.clamp(floor_probs, 0, 1)  # [batch_size] ensure floor probabilities within [0, 1]

    # === Ensure total probability is 1 ===
    total_probs = topk_probs.sum(dim=-1) + max(0, vocab_size_min - topk) * floor_probs  # [batch_size] total probs
    n_topk_probs = topk_probs / total_probs[:, None]  # [batch_size, topk] normalized topk_probs
    n_floor_probs = floor_probs / total_probs  # [batch_size] normalized floor_probs

    val_probs = torch.zeros(batch_size).to(topk_probs.device)  # accumulate probabilities when first tokens match
    match_probs = torch.zeros(batch_size).to(topk_probs.device)  # accumulate probabilities when sub target matches phrase
    for b in range(batch_size):
        target_phrase = target_phrases[b]
        if not isinstance(target_phrase, torch.Tensor):
            target_phrase = torch.tensor(target_phrases[b])
        if isinstance(target_phrase, torch.FloatTensor):
            target_phrase = target_phrase.round().int()

        match = (topk_tokens[b, :, 0] == target_phrase[0].item())  # bool where first tokens match (validation token)
        if match.sum() > 0:
            val_probs[b] = n_topk_probs[b, match].sum()  # accumulate all matches
        else:  # no matches
            val_probs[b] = n_floor_probs[b]  # assume match is in non-topk tokens with avg floor_prob

        # === Integrate sub target matches ===
        check_len = min(max_len - 1, len(target_phrase))
        for c in range(1, check_len + 1):  # progressively increase sub target length
            target = ignore_index * torch.ones(check_len, dtype=torch.int32).to(topk_tensor.device)  # [-100, ..., -100]
            target[:c] = target_phrase[:c]  # [tok0, tok1, ...tokc, -100, ..., -100]

            # Find sub target matches
            match = (topk_tokens[b, :, :check_len] == target)
            match_idx = torch.where(match.sum(dim=-1) == check_len)[0]  # phrase indices which match sub target

            if len(match_idx):  # at least one match
                match_probs[b] += n_topk_probs[b, match_idx].sum()  # accumulate all matches
            else:  # no matches
                match_probs[b] += n_floor_probs[b]  # assume match is in non-topk tokens with avg floor_prob

    val_probs = torch.clamp(val_probs, 0, 1)  # [batch_size] ensure 0 <= total probability <= 1
    loss_val = - torch.log(val_probs + 1e-40)  # [batch_size] calculate cross entropy loss

    match_probs = torch.clamp(match_probs, 0, 1)  # [batch_size] ensure 0 <= total probability <= 1
    loss = - torch.log(match_probs + 1e-40)  # [batch_size] calculate cross entropy loss

    if reduce:
        if not hasattr(loss_val, reduction) or not hasattr(loss, reduction):
            raise RuntimeError(f'phase_cross_entropy(): Reduction function {reduction} not found.')
        loss_val = getattr(loss_val, reduction)()
        loss = getattr(loss, reduction)()
        if loss.numel() > 1:
            raise ValueError(f'phase_cross_entropy(): Expected reduction to scalar, obtained {loss.shape} instead.')

    return loss_val, loss

