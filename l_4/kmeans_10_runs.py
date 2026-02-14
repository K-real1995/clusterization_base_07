# -*- coding: utf-8 -*-
"""
Эвристика с несколькими запусками K-means.

  1. 10 запусков K-means с k=2.
  2. В каждом запуске сохраняем координаты центроидов в общий массив c.
  3. Визуализируем: исходные точки фоном, точки из c поверх.
"""

import sys
import os
_HW_DIR = os.path.dirname(os.path.abspath(__file__))
if _HW_DIR not in sys.path:
    sys.path.insert(0, _HW_DIR)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from config import FIGURES_DIR
from data_loader import load_clustering_data

os.makedirs(FIGURES_DIR, exist_ok=True)

X = load_clustering_data()
all_centers = []
for run in range(10):
    model = KMeans(n_clusters=2, random_state=run, n_init=1)
    model.fit(X)
    all_centers.append(model.cluster_centers_)
# c — массив формы (20, 2): в каждом из 10 запусков по 2 центроида.
c = np.vstack(all_centers)
n_centroids_total = len(c)
print(f"Всего точек центроидов: {n_centroids_total} (2 центроида × 10 запусков)")

# Номер запуска для каждой точки: [0,0, 1,1, ..., 9,9] — чтобы раскрасить по запускам.
run_ids = np.repeat(np.arange(10), 2)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X[:, 0], X[:, 1], c="lightgray", alpha=0.5, s=25, label="Исходные точки")
scatter_cent = ax.scatter(
    c[:, 0], c[:, 1],
    c=run_ids,
    cmap="tab10",
    marker="o",
    s=70,
    alpha=0.85,
    edgecolors="black",
    linewidths=1,
    label=f"Центроиды ({n_centroids_total} точек: 10 запусков × k=2)",
)
ax.set_xlabel("Координата X")
ax.set_ylabel("Координата Y")
ax.set_title("Центроиды 10 запусков K-means (k=2) на фоне данных")
plt.colorbar(scatter_cent, ax=ax, label="Номер запуска (0–9)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
out_path = os.path.join(FIGURES_DIR, "kmeans_10_runs_centroids.png")
plt.savefig(out_path, dpi=120)
plt.show()
print("График сохранён:", out_path)
