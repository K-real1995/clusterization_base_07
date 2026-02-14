# -*- coding: utf-8 -*-
"""
K-means с 3 кластерами и визуализация.

  1. Загрузить датасет из data/clustering.pkl (в корне проекта).
  2. Обучить KMeans для n_clusters=3.
  3. Визуализировать: точки по кластерам, центроиды отдельно.
"""

import sys
import os
_HW_DIR = os.path.dirname(os.path.abspath(__file__))
if _HW_DIR not in sys.path:
    sys.path.insert(0, _HW_DIR)

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from config import FIGURES_DIR
from data_loader import load_clustering_data

# Папка для графиков — создаём, если нет.
os.makedirs(FIGURES_DIR, exist_ok=True)

X = load_clustering_data()
model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X)
labels = model.labels_
centers = model.cluster_centers_

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    X[:, 0], X[:, 1],
    c=labels,
    cmap="viridis",
    alpha=0.6,
    edgecolors="k",
    linewidths=0.3,
    s=30,
)
ax.scatter(
    centers[:, 0], centers[:, 1],
    c="red",
    marker="X",
    s=200,
    edgecolors="black",
    linewidths=2,
    label="Центроиды",
)
ax.set_xlabel("Координата X")
ax.set_ylabel("Координата Y")
ax.set_title("K-means, 3 кластера (датасет clustering.pkl)")
ax.legend()
plt.colorbar(scatter, ax=ax, label="Номер кластера")
plt.tight_layout()
out_path = os.path.join(FIGURES_DIR, "kmeans_3_clusters.png")
plt.savefig(out_path, dpi=120)
plt.show()
print("График сохранён:", out_path)
