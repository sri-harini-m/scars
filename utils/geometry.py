#utils/geometry.py

import torch 

def l2_distance(model_a, model_b):
    distance = 0
    with torch.no_grad():
        for p1, p2 in zip(model_a.parameters(), model_b.parameters()):
            distance +=  (p1-p2).pow(2).sum()
    return distance.sqrt().item()
