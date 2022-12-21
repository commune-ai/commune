import torch
def torch_batchdictlist2dict(batch_dict_list, dim=0):
    """
    converts
        batch_dict_list: dictionary (str, tensor)
        to
        out_batch_dict : dictionary (str,tensor)

    along dimension (dim)

    """
    out_batch_dict = {}
    for batch_dict in batch_dict_list:

        for k, v in batch_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            if k in out_batch_dict:
                out_batch_dict[k].append(v)
            else:
                out_batch_dict[k] = [v]

    # stack
    return {k: torch.cat(v, dim=dim) for k, v in out_batch_dict.items()}


def tensor_dict_shape(input_dict):
    out_dict = {}

    """should only have tensors/np.arrays in leafs"""
    for k,v in input_dict.items():
        if isinstance(v,dict):
            out_dict[k] = tensor_dict_shape(v)
        elif type(v) in [torch.Tensor, np.ndarray]:
            out_dict[k] = v.shape

    return out_dict


def check_distributions(kwargs):
    return {k: {"mean": round(v.double().mean().item(), 2), "std": round(v.double().std().item(), 2)} for k, v in
            kwargs.items() if isinstance(v, torch.Tensor)}




def confuse_gradients(model):
    """

    :param model: model
    :return:
    """
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data = torch.randn(p.grad.data.shape).to(p.grad.data.device)

