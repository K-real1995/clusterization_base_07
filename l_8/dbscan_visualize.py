# -*- coding: utf-8 -*-
"""
DBSCAN с min_samples=8 и визуализация результата.

  - Загрузка датасета из data/clustering.pkl
  - Масштабирование признаков (StandardScaler) для стабильной работы DBSCAN
  - Обучение DBSCAN(eps=0.5, min_samples=8)
  - Визуализация: кластеры разными цветами, шум (noise) отдельно
"""

import sys
import os

_HW_DIR = os.path.dirname(os.path.abspath(__file__))
if _HW_DIR not in sys.path:
    sys.path.insert(0, _HW_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from config import FIGURES_DIR
from data_loader import load_clustering_data

os.makedirs(FIGURES_DIR, exist_ok=True)

# Загрузка данных
X_raw = load_clustering_data()

# Масштабирование для DBSCAN (рекомендуемая практика)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# Обучение DBSCAN: min_samples=8, eps подбирается под масштаб после StandardScaler
model = DBSCAN(eps=0.5, min_samples=8)
labels = model.fit_predict(X)

n_clusters = len(set(labels) - {-1})
n_noise = int(np.sum(labels == -1))

# Визуализация в исходных координатах для интерпретации (опционально можно рисовать X)
# Рисуем в масштабированных координатах, т.к. кластеры определены в этом пространстве
fig, ax = plt.subplots(figsize=(8, 6))

# Палитра: кластеры — цветные, шум — серый
unique_labels = sorted(set(labels))
colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels) - 1, 1)))

for i, label in enumerate(unique_labels):
    if label == -1:
        color = (0.4, 0.4, 0.4, 0.7)
        marker = "x"
        size = 25
        legend_label = "Шум (noise)"
        edgecolors = "none"
        linewidths = 0
    else:
        color = colors[i % len(colors)]
        marker = "o"
        size = 40
        legend_label = f"Кластер {label}"
        edgecolors = "k"
        linewidths = 0.3
    mask = labels == label
    ax.scatter(
        X[mask, 0],
        X[mask, 1],
        c=[color],
        marker=marker,
        s=size,
        edgecolors=edgecolors,
        linewidths=linewidths,
        label=legend_label,
    )

ax.set_xlabel("Признак 1 (нормализовано)")
ax.set_ylabel("Признак 2 (нормализовано)")
ax.set_title(f"DBSCAN (min_samples=8, eps=0.5)\nКластеров: {n_clusters}, шум: {n_noise} точек")
ax.legend(loc="best", fontsize=8)
ax.set_aspect("equal")
plt.tight_layout()

out_path = os.path.join(FIGURES_DIR, "dbscan_min_samples_8.png")
plt.savefig(out_path, dpi=120)
plt.close()

print("График сохранён:", out_path)
print(f"Кластеров: {n_clusters}, точек шума: {n_noise}")
