#scripts/weight_distance.py

from utils.geometry import l2_distance
from utils.model_utils import load_model

m0, _ = load_model("checkpoints/original")
mr, _ = load_model("checkpoints/recovered/final")
mf, _ = load_model("checkpoints/control/final")

print("L2(theta0, theta_r)  [ROB-ish]:", l2_distance(m0, mr))
print("L2(theta0, theta_ft) [control]:", l2_distance(m0, mf))