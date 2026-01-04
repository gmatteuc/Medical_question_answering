"""
Medical QA - Relevance Scoring Logic Test
=============================================

This script tests the "Relevance" metric of the LLM-as-a-Judge.
It uses controlled test cases (Fully Relevant, Mostly Relevant, Partially Relevant, Irrelevant)
to verify that the judge assigns scores consistent with the defined rubric.
"""

import sys
import os
import torch
import pandas as pd

# Add project root to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag import RAGPipeline
from src.evaluation import evaluate_rag_response

def run_relevance_scoring_test() -> None:
    """
    Runs the relevance scoring logic test suite.
    Iterates through predefined test cases and compares the judge's score with the expected score.
    """
    print("Initializing RAG Pipeline for Judge...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rag = RAGPipeline(device=device)
    rag.load_resources()
    
    print("\n=== STARTING RELEVANCE SCORING LOGIC TEST ===\n")
    
    test_cases = [
        {
            "name": "Fully Relevant (Control)",
            "question": "What are the symptoms of COVID-19?",
            "answer": "The symptoms include fever, cough, and fatigue.",
            "expected": 5.0
        },
        {
            "name": "Mostly Relevant (Target 4.0)",
            "question": "What are the symptoms and treatments of COVID-19?",
            "answer": "The symptoms include fever and cough. (Missed treatment part)",
            "expected": 4.0
        },
        {
            "name": "Partially Relevant (Target 3.0)",
            "question": "How do I treat a headache?",
            "answer": "Headaches are very painful and can be caused by stress.",
            "expected": 3.0
        },
        {
            "name": "Irrelevant (Control)",
            "question": "What is the capital of France?",
            "answer": "The mitochondria is the powerhouse of the cell.",
            "expected": 1.0
        }
    ]
    
    for case in test_cases:
        print(f"--- Testing Case: {case['name']} ---")
        print(f"Question: {case['question']}")
        print(f"Answer:   {case['answer']}")
        
        # Evaluate
        # Note: evaluate_rag_response expects a list of source dicts for context, but Relevance ignores context
        sources = [] 
        result = evaluate_rag_response(rag, case['question'], case['answer'], sources)
        
        score = result['Relevance']
        reasoning = result['Relevance_Reasoning']
        
        print(f"SCORE: {score}")
        print(f"REASONING: {reasoning}")
        print(f"Expected roughly: {case['expected']}")
        
        if abs(score - case['expected']) <= 0.5:
            print(">>> RESULT: MATCH")
        else:
            print(">>> RESULT: MISMATCH")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    run_relevance_scoring_test()
