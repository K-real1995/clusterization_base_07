# -*- coding: utf-8 -*-
"""
L_10_2 — K-means с k=4 и метрика AMI (Adjusted Mutual Information).

Задача:
  - Обучить k-means для k=4 на датасете data/clustering.pkl.
  - Сравнить метрику AMI при k=2 и k=4: стало лучше или хуже?
  - Ответить: какое количество кластеров лучше по метрике AMI?

AMI (Adjusted Mutual Information):
  - Используется, когда у нас ЕСТЬ истинные метки кластеров (ground truth).
  - Показывает, насколько предсказанные кластеры совпадают с истинными,
    с поправкой на случайное совпадение (значение 0 — как случайное разбиение, 1 — полное совпадение).
  - В отличие от силуэтта, AMI требует наличия эталонной разметки в данных.
"""

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

from config import CLUSTERING_PKL_PATH
from data_loader import load_clustering_data

# =============================================================================
# 1. Загрузка данных (с истинными метками для AMI)
# =============================================================================
X, y_true = load_clustering_data(return_labels=True)
print(f"Загружено объектов: {X.shape[0]}, признаков: {X.shape[1]}")
print(f"Путь к данным: {CLUSTERING_PKL_PATH}")

if y_true is None:
    print("\nВ датасете нет истинных меток (y/labels). Для расчёта AMI нужны эталонные метки.")
    print("Запускаем k-means и выводим только предсказанные метки (AMI вычислить нельзя).")
else:
    print(f"Истинные метки: найдено {len(np.unique(y_true))} уникальных классов.\n")

# =============================================================================
# 2. K-means с k=2
# =============================================================================
kmeans_k2 = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_k2.fit(X)
labels_k2 = kmeans_k2.labels_

# AMI сравнивает предсказанные метки с истинными (если они есть)
if y_true is not None:
    ami_k2 = adjusted_mutual_info_score(y_true, labels_k2)
    print(f"K-means, k=2:")
    print(f"  AMI (Adjusted Mutual Information): {ami_k2:.4f}\n")
else:
    ami_k2 = None

# =============================================================================
# 3. K-means с k=4 (как в задании)
# =============================================================================
kmeans_k4 = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_k4.fit(X)
labels_k4 = kmeans_k4.labels_

if y_true is not None:
    ami_k4 = adjusted_mutual_info_score(y_true, labels_k4)
    print(f"K-means, k=4:")
    print(f"  AMI (Adjusted Mutual Information): {ami_k4:.4f}\n")
else:
    ami_k4 = None

# =============================================================================
# 4. Сравнение и вывод
# =============================================================================
if y_true is not None and ami_k2 is not None and ami_k4 is not None:
    print("--- Сравнение ---")
    print(f"  AMI k=2: {ami_k2:.4f}")
    print(f"  AMI k=4: {ami_k4:.4f}")

    if ami_k4 > ami_k2:
        print("  При переходе с k=2 на k=4 метрика AMI ВЫРОСЛА -> стало ЛУЧШЕ.")
        print("  По метрике AMI лучше количество кластеров k=4.")
    elif ami_k4 < ami_k2:
        print("  При переходе с k=2 на k=4 метрика AMI УПАЛА -> стало ХУЖЕ.")
        print("  По метрике AMI лучше количество кластеров k=2.")
    else:
        print("  Метрики AMI совпали.")

    print("\nИтог: какое k лучше по AMI? -", "k=4" if ami_k4 > ami_k2 else "k=2")
else:
    print("Итог: для вывода сравнения по AMI добавьте в data/clustering.pkl ключ 'y' (или 'labels') с истинными метками.")
