# -*- coding: utf-8 -*-
"""
Конфигурация путей к данным для L_8.
"""

import os

# Корень проекта — родитель папки l_8
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CLUSTERING_PKL_PATH = os.path.join(DATA_DIR, "clustering.pkl")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(THIS_DIR, "figures")
