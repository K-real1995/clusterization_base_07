# -*- coding: utf-8 -*-
"""
Конфигурация путей для задания 10.3.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CLUSTERING_HW_CSV_PATH = os.path.join(DATA_DIR, "clustering.csv")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
