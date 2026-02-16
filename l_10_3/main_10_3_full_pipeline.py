# -*- coding: utf-8 -*-
"""
Задание 10.3 — Полный пайплайн по кластеризации на датасете clustering.csv.

В рамках домашней работы:
  1. Загрузить датасет data/clustering_hw.csv и визуализировать данные.
  2. Вычислить оптимальное количество кластеров k (например, по силуэтту или методу локтя).
  3. Обучить k-means с найденным k.
  4. Определить: к какому кластеру относятся точки (5, 8) и (0, 5)? Один кластер или разные?
  5. Оценить качество кластеризации по AMI (если в данных есть истинные метки).
"""

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Неинтерактивный бэкенд: графики сохраняются в файлы и не блокируют консоль при plt.show()
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_mutual_info_score

# Пробуем pandas для чтения CSV; если нет — читаем через csv/numpy
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    import csv

from config import CLUSTERING_HW_CSV_PATH, FIGURES_DIR

os.makedirs(FIGURES_DIR, exist_ok=True)

# =============================================================================
# 1. Загрузка данных из CSV
# =============================================================================
def load_clustering_hw():
    """
    Загружает data/clustering.csv.
    Ожидаемые колонки: две числовые (признаки) и опционально колонка с истинными метками.
    Возвращает: X (признаки), y_true (метки или None), имена колонок признаков.
    """
    if not os.path.isfile(CLUSTERING_HW_CSV_PATH):
        # Генерируем тестовый датасет для проверки кода без файла
        rng = np.random.RandomState(42)
        c1 = rng.randn(80, 2) + [3, 6]
        c2 = rng.randn(80, 2) + [1, 2]
        X = np.vstack([c1, c2])
        y_true = np.array([0] * 80 + [1] * 80)
        return X, y_true, ["x", "y"]

    if HAS_PANDAS:
        df = pd.read_csv(CLUSTERING_HW_CSV_PATH)
        # Ищем колонки с признаками: часто x, y или 0, 1, или первые две числовые
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            raise ValueError("В CSV нужно минимум 2 числовые колонки.")
        # Признаки — первые две числовые (или с именами x, y)
        feat_cols = [c for c in ["x", "y", "X", "Y", "0", "1"] if c in df.columns]
        if len(feat_cols) >= 2:
            col1, col2 = feat_cols[0], feat_cols[1]
        else:
            col1, col2 = numeric.columns[0], numeric.columns[1]
        X = df[[col1, col2]].values
        # Истинные метки — если есть колонка label/cluster/target и т.д.
        y_true = None
        for key in ("label", "cluster", "target", "y", "class"):
            if key in df.columns:
                y_true = df[key].values.ravel()
                break
        return X, y_true, [col1, col2]

    # Без pandas: читаем CSV вручную
    with open(CLUSTERING_HW_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    # Предполагаем первые две колонки — числа (признаки)
    data = []
    for row in rows:
        if len(row) >= 2:
            try:
                data.append([float(row[0]), float(row[1])])
            except ValueError:
                continue
    X = np.array(data)
    y_true = None
    if len(header) >= 3 and header[2].lower() in ("label", "cluster", "target", "y"):
        col_idx = 2
        y_true = np.array([row[col_idx] if len(row) > col_idx else 0 for row in rows], dtype=int)
    return X, y_true, header[:2]


X, y_true, feat_names = load_clustering_hw()
print(f"Загружено объектов: {X.shape[0]}, признаков: {X.shape[1]}")
print(f"Путь к данным: {CLUSTERING_HW_CSV_PATH}")
if y_true is not None:
    print(f"Истинные метки: {len(np.unique(y_true))} классов.\n")
else:
    print("Истинные метки в файле не найдены (AMI можно будет не считать).\n")

# =============================================================================
# 2. Визуализация данных (как в задании — scatter по двум осям)
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
# Если есть истинные метки — раскрашиваем по ним; иначе все точки одним цветом
if y_true is not None:
    scatter = ax.scatter(
        X[:, 0], X[:, 1],
        c=y_true,
        cmap="viridis",
        alpha=0.7,
        edgecolors="k",
        linewidths=0.3,
        s=40,
    )
    plt.colorbar(scatter, ax=ax, label="Класс (истинные метки)")
else:
    ax.scatter(X[:, 0], X[:, 1], alpha=0.7, s=40, c="steelblue", edgecolors="k", linewidths=0.3)
ax.set_xlabel(feat_names[0] if feat_names else "X")
ax.set_ylabel(feat_names[1] if len(feat_names) > 1 else "Y")
ax.set_title("Датасет clustering_hw.csv — визуализация данных")
plt.tight_layout()
path_vis = os.path.join(FIGURES_DIR, "clustering_data.png")
plt.savefig(path_vis, dpi=120)
plt.close()
print(f"Визуализация сохранена: {path_vis}\n")

# =============================================================================
# 3. Подбор оптимального количества кластеров k
# =============================================================================
# Перебираем k от 2 до 10 (или до min(10, n_samples//2)), считаем силуэт.
# Оптимальное k — то, при котором силуэт максимален (или по «локтю» на графике инерции).
K_MAX = min(10, X.shape[0] // 2, X.shape[0] - 1)
K_MAX = max(2, K_MAX)
k_range = range(2, K_MAX + 1)
silhouette_scores = []
inertias = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    silhouette_scores.append(silhouette_score(X, km.labels_, metric="euclidean"))
    inertias.append(km.inertia_)

# Оптимальное k по силуэтту
best_k_sil = k_range[np.argmax(silhouette_scores)]
best_sil = max(silhouette_scores)
print(f"Оптимальное k по метрике силуэтта: {best_k_sil} (Silhouette = {best_sil:.4f})")
print("Силуэт по k:", dict(zip(k_range, [f"{s:.3f}" for s in silhouette_scores])))

# Рекомендуем использовать это k для k-means
optimal_k = best_k_sil
print(f"\nВыбрано количество кластеров k = {optimal_k}\n")

# =============================================================================
# 4. Обучение K-means с выбранным k
# =============================================================================
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(X)
labels_pred = kmeans.labels_
centers = kmeans.cluster_centers_

# Визуализация результата кластеризации
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    X[:, 0], X[:, 1],
    c=labels_pred,
    cmap="viridis",
    alpha=0.7,
    edgecolors="k",
    linewidths=0.3,
    s=40,
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
ax.set_xlabel(feat_names[0] if feat_names else "X")
ax.set_ylabel(feat_names[1] if len(feat_names) > 1 else "Y")
ax.set_title(f"K-means, k={optimal_k} (clustering.csv)")
ax.legend()
plt.colorbar(scatter, ax=ax, label="Предсказанный кластер")
plt.tight_layout()
path_clusters = os.path.join(FIGURES_DIR, "clustering_kmeans.png")
plt.savefig(path_clusters, dpi=120)
plt.close()
print(f"График кластеризации сохранён: {path_clusters}\n")

# =============================================================================
# 5. К какому кластеру относятся точки (5, 8) и (0, 5)?
# =============================================================================
# predict() принимает массив форм (n_samples, n_features) — одна или несколько точек
point_a = np.array([[5.0, 8.0]])
point_b = np.array([[0.0, 5.0]])
cluster_a = kmeans.predict(point_a)[0]
cluster_b = kmeans.predict(point_b)[0]

print("--- Принадлежность точек к кластерам ---")
print(f"  Точка x = (5, 8)  -> кластер {cluster_a}")
print(f"  Точка x = (0, 5)  -> кластер {cluster_b}")
if cluster_a == cluster_b:
    print("  Они принадлежат ОДНОМУ кластеру.")
else:
    print("  Они принадлежат РАЗНЫМ кластерам.")

# =============================================================================
# 6. Качество кластеризации по AMI
# =============================================================================
if y_true is not None:
    ami = adjusted_mutual_info_score(y_true, labels_pred)
    print(f"\nКачество кластеризации по AMI: {ami:.4f}")
else:
    print("\nИстинных меток нет — AMI не вычисляется.")
    ami = None

print("\nГотово.")
