#utils/geometry.py

import torch 

def l2_distance(model_a, model_b):
    distance = 0
    n_params = 0
    with torch.no_grad():
        for p1, p2 in zip(model_a.parameters(), model_b.parameters()):
            diff = (p1 - p2).float()
            distance += diff.pow(2).sum()
            n_params += p1.numel()
    return distance.sqrt().item(), n_params

def rms_distance(model_a, model_b):
    sq_sum = 0
    n_params = 0
    with torch.no_grad():
        for p1, p2 in zip(model_a.parameters(), model_b.parameters()):
            diff = (p1 - p2).float()
            sq_sum += diff.pow(2).sum().item()
            n_params += p1.numel()
    return (sq_sum / n_params) ** 0.5


def layerwise_distances(model_a, model_b):
    results = {}
    with torch.no_grad():
        for (name, p1), (_, p2) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            diff = (p1 - p2).float()
            rms = (diff.pow(2).mean()).sqrt().item()
            results[name] = rms
    return results
