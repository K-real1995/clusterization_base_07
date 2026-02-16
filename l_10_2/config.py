# -*- coding: utf-8 -*-
"""
Конфигурация путей для L_10_2.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CLUSTERING_PKL_PATH = os.path.join(DATA_DIR, "clustering.pkl")
