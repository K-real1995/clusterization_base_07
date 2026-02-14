"""
Конфигурация путей к данным.
"""

import os

# Корень проекта — родитель папки lesson4_hw.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Папка с данными в корне проекта (как в Colab: data/clustering.pkl).
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CLUSTERING_PKL_PATH = os.path.join(DATA_DIR, "clustering.pkl")

# Папка для сохранения графиков
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(THIS_DIR, "figures")
