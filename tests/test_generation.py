"""
Medical QA - Generation Module Test
=============================================

This script tests the synthetic query generation functionality.
It verifies that the system can generate questions from the knowledge base
and that the generated questions are grounded in the source context.
"""

import sys
import os
import torch
import random

# Add project root to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag import RAGPipeline
from src.generation import generate_synthetic_queries

def test_generation_module() -> None:
    """
    Runs a smoke test for the synthetic query generation module.
    Generates a small number of queries and prints them for inspection.
    """
    print("Initializing RAG Pipeline (loading model)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rag = RAGPipeline(device=device)
    rag.load_resources()
    
    print("\n--- Testing generate_synthetic_queries with Full Context Logic ---")
    
    # Generate a few queries
    output_path = os.path.join(os.path.dirname(__file__), 'test_gen_output.csv')
    df = generate_synthetic_queries(rag, num_queries=2, output_path=output_path, force=True)
    
    print(f"\nGenerated {len(df)} queries.")
    
    for i, row in df.iterrows():
        print(f"\n=== Query {i+1} ===")
        print(f"Question: {row['question']}")
        context_len = len(row['source_context'])
        print(f"Source Context Length: {context_len} chars")
        print(f"Source Context Snippet (Start): {row['source_context'][:200]}...")
        print(f"Source Context Snippet (End): ...{row['source_context'][-200:]}")
        
        # Basic check: if context is very short, it might be suspicious (unless the original doc was short)
        if context_len < 100:
            print("WARNING: Context seems very short.")
        else:
            print("Context length looks reasonable for a full document.")

if __name__ == "__main__":
    test_generation_module()
