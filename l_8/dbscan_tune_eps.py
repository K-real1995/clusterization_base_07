# -*- coding: utf-8 -*-
"""
Подбор eps для DBSCAN по k-distance (k=min_samples=8) и сравнение вариантов.

  1. Строит график k-distance: расстояние до 8-го соседа (отсортировано).
  2. Предлагает eps по "колену" кривой (резкий рост = граница плотности).
  3. Запускает DBSCAN с несколькими eps и сохраняет визуализации для сравнения.
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
from sklearn.neighbors import NearestNeighbors

from config import FIGURES_DIR
from data_loader import load_clustering_data

os.makedirs(FIGURES_DIR, exist_ok=True)

MIN_SAMPLES = 8

# Загрузка и масштабирование
X_raw = load_clustering_data()
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
n_points = X.shape[0]

# --- K-distance для k=min_samples ---
knn = NearestNeighbors(n_neighbors=MIN_SAMPLES)
knn.fit(X)
distances, _ = knn.kneighbors(X)
# Расстояние до k-го соседа (индекс MIN_SAMPLES-1, т.к. 0 — сам объект)
k_dist = np.sort(distances[:, MIN_SAMPLES - 1])

# График k-distance: ищем "колено" (резкий рост)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(len(k_dist)), k_dist, "b-", linewidth=1.5)
ax.set_xlabel("Точки (отсортированы по k-distance)")
ax.set_ylabel("Расстояние до 8-го соседа")
ax.set_title("K-distance (k=8) для подбора eps")
ax.grid(True, alpha=0.3)

# Эвристика eps: точка максимальной кривизны (второй разности)
# Упрощённо — берём перцентиль, где кривая уже пошла вверх (например 85–92%)
idx_90 = int(0.90 * len(k_dist))
idx_85 = int(0.85 * len(k_dist))
eps_90 = float(k_dist[idx_90])
eps_85 = float(k_dist[idx_85])
# Классическое "колено": максимум второй разности (ускорения)
d2 = np.diff(k_dist, n=2)
knee_idx = np.argmax(d2) + 1 if len(d2) > 0 else len(k_dist) // 2
eps_knee = float(k_dist[min(knee_idx, len(k_dist) - 1)])

ax.axhline(eps_knee, color="green", linestyle="--", alpha=0.8, label=f"eps (колено) ≈ {eps_knee:.3f}")
ax.axhline(eps_90, color="orange", linestyle=":", alpha=0.8, label=f"eps (90% перц.) ≈ {eps_90:.3f}")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "dbscan_kdistance_k8.png"), dpi=120)
plt.close()
print("K-distance график сохранён: figures/dbscan_kdistance_k8.png")
print(f"  Предложенный eps (колено): {eps_knee:.3f}")
print(f"  eps (90% перцентиль):     {eps_90:.3f}")

# Подбор eps: ищем значение с 3 кластерами и минимальным шумом (или 2 кластера, если 3 недостижимо)
best_eps, best_noise, best_n_clusters = 0.5, n_points, 0
for eps_try in np.linspace(0.35, 0.85, 51):
    lab = DBSCAN(eps=float(eps_try), min_samples=MIN_SAMPLES).fit_predict(X)
    n_c = len(set(lab) - {-1})
    n_n = int(np.sum(lab == -1))
    if n_c >= 2 and (n_c > best_n_clusters or (n_c == best_n_clusters and n_n < best_noise)):
        best_eps, best_noise, best_n_clusters = eps_try, n_n, n_c
eps_suggested = round(best_eps, 3)
print(f"  Podbor: eps ~ {eps_suggested}, klasterov: {best_n_clusters}, shum: {best_noise}")

# Варианты для сравнения
candidates = [
    ("eps=0.50 (исходный)", 0.50),
    ("eps подобранный (баланс)", eps_suggested),
    ("eps=0.40 (сильнее разделить)", 0.40),
    ("eps=0.88 (90% перц., меньше шума)", eps_90),
]

def plot_dbscan(X, labels, title, path):
    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels) - 1, 1)))
    for i, label in enumerate(unique_labels):
        if label == -1:
            color = (0.4, 0.4, 0.4, 0.7)
            marker, size = "x", 25
            legend_label = "Шум (noise)"
            edgecolors, linewidths = "none", 0
        else:
            color = colors[i % len(colors)]
            marker, size = "o", 40
            legend_label = f"Кластер {label}"
            edgecolors, linewidths = "k", 0.3
        mask = labels == label
        kwargs = dict(c=[color], marker=marker, s=size, label=legend_label)
        if marker != "x":
            kwargs["edgecolors"] = edgecolors
            kwargs["linewidths"] = linewidths
        ax.scatter(X[mask, 0], X[mask, 1], **kwargs)
    ax.set_xlabel("Признак 1 (нормализовано)")
    ax.set_ylabel("Признак 2 (нормализовано)")
    ax.set_title(f"{title}\nКластеров: {n_clusters}, шум: {n_noise} точек")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return n_clusters, n_noise

print("\nЗапуск DBSCAN с разными eps:")
for name, eps in candidates:
    eps = float(eps)
    model = DBSCAN(eps=eps, min_samples=MIN_SAMPLES)
    labels = model.fit_predict(X)
    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    fname = f"dbscan_{safe_name}.png"
    path = os.path.join(FIGURES_DIR, fname)
    plot_dbscan(X, labels, f"DBSCAN min_samples={MIN_SAMPLES}, eps={eps:.3f}", path)
    print(f"  {name}: eps={eps:.3f} -> кластеров: {n_clusters}, шум: {n_noise}")

# Итоговый вариант с подобранным eps
model_best = DBSCAN(eps=eps_suggested, min_samples=MIN_SAMPLES)
labels_best = model_best.fit_predict(X)
plot_dbscan(
    X, labels_best,
    f"DBSCAN (подобранный eps={eps_suggested:.3f}, min_samples={MIN_SAMPLES})",
    os.path.join(FIGURES_DIR, "dbscan_tuned.png"),
)
print("\nИтоговый вариант (подобранный eps) сохранён: figures/dbscan_tuned.png")
