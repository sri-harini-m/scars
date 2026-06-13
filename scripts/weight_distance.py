from utils.geometry import l2_distance
from utils.model_utils import load_model

m0, _ = load_model("checkpoints/original")
mr, _ = load_model("checkpoints/recovered/final")

print(l2_distance(m0, mr))