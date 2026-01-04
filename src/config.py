"""
Medical QA - Configuration Module
=============================================

This module defines the global configuration constants for the project.
It includes paths, model names, and RAG parameters.
"""

import os

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'medDataset_processed.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'knowledge_base.parquet')
INDEX_PATH = os.path.join(DATA_DIR, 'faiss_index.bin')

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_NAME = "google/gemma-2-2b-it"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# =============================================================================
# RAG Parameters
# =============================================================================
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5

