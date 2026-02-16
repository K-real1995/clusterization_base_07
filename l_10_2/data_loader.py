# -*- coding: utf-8 -*-
"""
Загрузка data/clustering.pkl с возможностью получить истинные метки (для AMI).
Во многих учебных датасетах в pkl лежит словарь с ключами 'X' и 'y' (или 'labels').
"""

import pickle
import warnings
import numpy as np

from config import CLUSTERING_PKL_PATH


def load_clustering_data(return_labels=False):
    """
    Загружает данные из data/clustering.pkl.

    Args:
        return_labels : bool
            Если True и в файле есть метки (y/labels/target), вернёт (X, y_true).
            Иначе вернёт только X (или (X, None)).

    Returns:
        X : np.ndarray
        y_true : np.ndarray или None — истинные метки кластеров (только при return_labels=True).
    """
    try:
        with open(CLUSTERING_PKL_PATH, "rb") as f:
            with warnings.catch_warnings():
                try:
                    from numpy.exceptions import VisibleDeprecationWarning
                    warnings.simplefilter("ignore", VisibleDeprecationWarning)
                except ImportError:
                    pass
                data = pickle.load(f)

        if isinstance(data, np.ndarray):
            return (data, None) if return_labels else data

        if isinstance(data, dict):
            # Ищем матрицу объектов
            X = None
            for key in ("X", "data", "points"):
                if key in data:
                    X = np.asarray(data[key])
                    break
            if X is None:
                X = np.asarray(list(data.values())[0])

            # Ищем истинные метки (для AMI)
            y_true = None
            if return_labels:
                for key in ("y", "labels", "target", "ground_truth"):
                    if key in data:
                        y_true = np.asarray(data[key]).ravel()
                        break

            return (X, y_true) if return_labels else X

        return (np.asarray(data), None) if return_labels else np.asarray(data)

    except FileNotFoundError:
        # Тестовые данные с известными метками (2 класса по 50 точек)
        rng = np.random.RandomState(42)
        c1 = rng.randn(50, 2) + [2, 2]
        c2 = rng.randn(50, 2) + [-1, -1]
        X = np.vstack([c1, c2])
        y_true = np.array([0] * 50 + [1] * 50) if return_labels else None
        return (X, y_true) if return_labels else X
