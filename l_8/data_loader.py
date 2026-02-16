# -*- coding: utf-8 -*-
"""
Загрузка данных для заданий по кластеризации (L_8).
"""

import pickle
import warnings
import numpy as np

from config import CLUSTERING_PKL_PATH


def load_clustering_data():
    """
    Загружает массив точек из data/clustering.pkl (в корне проекта).

    Returns:
        X : np.ndarray, shape (n_samples, 2) — координаты точек для кластеризации.
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
            return data
        if isinstance(data, dict):
            for key in ("X", "data", "points"):
                if key in data:
                    return np.asarray(data[key])
            return np.asarray(list(data.values())[0])
        return np.asarray(data)
    except FileNotFoundError:
        rng = np.random.RandomState(42)
        c1 = rng.randn(50, 2) + [2, 2]
        c2 = rng.randn(50, 2) + [-1, -1]
        c3 = rng.randn(50, 2) + [0, 2]
        return np.vstack([c1, c2, c3])
