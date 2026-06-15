#scripts/weight_distance.py

from utils.geometry import rms_distance, layerwise_distances
from utils.model_utils import load_model
import json, os

m0, _ = load_model("checkpoints/original")
mr, _ = load_model("checkpoints/recovered")
mf, _ = load_model("checkpoints/control")

print(f"RMS(theta0, theta_r)  [recovered]: {rms_distance(m0, mr):.6f}")
print(f"RMS(theta0, theta_ft) [control]:   {rms_distance(m0, mf):.6f}")

lw_r = layerwise_distances(m0, mr)
lw_ft = layerwise_distances(m0, mf)

os.makedirs("results/geometry", exist_ok=True)
with open("results/geometry/layerwise_recovered.json", "w") as f:
    json.dump(lw_r, f, indent=2)
with open("results/geometry/layerwise_control.json", "w") as f:
    json.dump(lw_ft, f, indent=2)

print("\nTop 10 layers with largest RMS scar (theta_r - theta_0):")
sorted_layers = sorted(lw_r.items(), key=lambda x: x[1], reverse=True)
for name, val in sorted_layers[:10]:
    print(f"  {name}: {val:.6f}")