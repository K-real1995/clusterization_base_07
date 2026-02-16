# -*- coding: utf-8 -*-
"""
L 10.1 — K-means с k=4 и метрика силуэтта (Silhouette).

Задача:
  - Обучить k-means для k=4 на датасете data/clustering.pkl.
  - Сравнить метрику силуэтта при k=2 и k=4: стало лучше или хуже?
  - Ответить: какое количество кластеров лучше — два или четыре?

Метрика силуэтта (Silhouette):
  - Используется, когда у нас НЕТ истинных меток кластеров (неконтролируемая оценка).
  - Для каждой точки считает: насколько она «похожа» на свой кластер vs на соседний.
  - Значения от -1 до 1: чем выше, тем лучше разделение на кластеры.
  - Среднее по всем точкам — silhouette_score в sklearn.
"""

import sys
import os

# Добавляем текущую папку в путь, чтобы импортировать config и data_loader
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import CLUSTERING_PKL_PATH
from data_loader import load_clustering_data

# =============================================================================
# 1. Загрузка данных
# =============================================================================
# X — матрица объектов (каждая строка — точка в 2D или многомерном пространстве)
X = load_clustering_data()
print(f"Загружено объектов: {X.shape[0]}, признаков: {X.shape[1]}")
print(f"Путь к данным: {CLUSTERING_PKL_PATH}\n")

# =============================================================================
# 2. K-means с k=2 (базовый вариант для сравнения)
# =============================================================================
# n_clusters=2 — разбиваем данные на 2 кластера
# random_state=42 — фиксируем случайность для воспроизводимости
# n_init=10 — алгоритм запускается 10 раз с разными начальными центроидами; выбирается лучший по инерции
kmeans_k2 = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_k2.fit(X)
labels_k2 = kmeans_k2.labels_

# Силуэт при k=2: передаём объекты X и предсказанные метки кластеров
silhouette_k2 = silhouette_score(X, labels_k2, metric="euclidean")
print(f"K-means, k=2:")
print(f"  Метрика силуэтта (Silhouette): {silhouette_k2:.4f}\n")

# =============================================================================
# 3. K-means с k=4 (как в задании)
# =============================================================================
kmeans_k4 = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_k4.fit(X)
labels_k4 = kmeans_k4.labels_

silhouette_k4 = silhouette_score(X, labels_k4, metric="euclidean")
print(f"K-means, k=4:")
print(f"  Метрика силуэтта (Silhouette): {silhouette_k4:.4f}\n")

# =============================================================================
# 4. Сравнение и вывод
# =============================================================================
print("--- Сравнение ---")
print(f"  Силуэт k=2: {silhouette_k2:.4f}")
print(f"  Силуэт k=4: {silhouette_k4:.4f}")

if silhouette_k4 > silhouette_k2:
    print("  При переходе с k=2 на k=4 метрика силуэтта ВЫРОСЛА -> стало ЛУЧШЕ по этой метрике.")
    print("  По метрике силуэтта лучше количество кластеров k=4.")
elif silhouette_k4 < silhouette_k2:
    print("  При переходе с k=2 на k=4 метрика силуэтта УПАЛА -> стало ХУЖЕ по этой метрике.")
    print("  По метрике силуэтта лучше количество кластеров k=2.")
else:
    print("  Метрики совпали (редко).")

print("\nИтог: какое k лучше по силуэтту? -", "k=4" if silhouette_k4 > silhouette_k2 else "k=2")
