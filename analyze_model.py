import joblib
import numpy as np

path = "C:\\Users\\LuisDiaz\\Documents\\axiom\\AxiomApi\\AxiomServiceIA\\models\\e9e3b15c-742b-4e47-930a-7db4f01e0105\\13.0.1\\model.pkl"
m = joblib.load(path)

kmeans = m["kmeans"]
hmm_model = m["hmm"]

print("=== KMeans ===")
print("Clusters:", kmeans.n_clusters)
print("Centroides:\n", np.round(kmeans.cluster_centers_, 4))

print("\n=== HMM ===")
print("Estados:", hmm_model.n_components)
print("Matriz de transición:\n", np.round(hmm_model.transmat_, 4))
print("Probabilidades iniciales:\n", np.round(hmm_model.startprob_, 4))
