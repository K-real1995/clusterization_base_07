# -*- coding: utf-8 -*-
"""
Конфигурация путей для l_10_1
Корень проекта — на уровень выше этой папки (L_10_1).
"""

import os

# Абсолютный путь к текущему файлу → папка L_10_1 → её родитель = корень проекта
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CLUSTERING_PKL_PATH = os.path.join(DATA_DIR, "clustering.pkl")
