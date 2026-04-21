import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
km = pd.read_csv("kmeans_results.csv")
kmd = pd.read_csv("kmedoids_results.csv")

# Keep only synthetic (for scaling analysis)
km_syn = km[km["dataset"] == "synthetic"]
kmd_syn = kmd[kmd["dataset"] == "synthetic"]

# Sort by n (important for correct line plots)
km_syn = km_syn.sort_values("n")
kmd_syn = kmd_syn.sort_values("n")

# =========================
# 📈 GRAPH 1: K-MEANS
# =========================
plt.plot(km_syn["n"], km_syn["avg_runtime"], marker='o')

plt.xlabel("Number of Points (n)")
plt.ylabel("Runtime (seconds)")
plt.title("K-Means Time Complexity Scaling")
plt.grid(True)
plt.show()

# =========================
# 📈 GRAPH 2: K-MEDOIDS
# =========================
plt.plot(kmd_syn["n"], kmd_syn["avg_runtime"], marker='o')

plt.xlabel("Number of Points (n)")
plt.ylabel("Runtime (seconds)")
plt.title("K-Medoids Time Complexity Scaling")
plt.grid(True)
plt.show()
