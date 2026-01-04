"""
Medical QA - Partial Scoring Logic Test
=============================================

This script tests the "Faithfulness" metric of the LLM-as-a-Judge.
It uses controlled test cases (Perfect Match, Minor Miss, Major Miss, Hallucination)
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

def run_partial_scoring_test() -> None:
    """
    Runs the partial scoring logic test suite.
    Iterates through predefined test cases and compares the judge's score with the expected score.
    """
    print("Initializing RAG Pipeline for Judge...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rag = RAGPipeline(device=device)
    rag.load_resources()
    
    print("\n=== STARTING PARTIAL SCORING LOGIC TEST ===\n")
    
    test_cases = [
        {
            "name": "Perfect Match (Control)",
            "context": "The primary symptoms of COVID-19 include fever, cough, fatigue, and loss of taste or smell.",
            "question": "What are the symptoms of COVID-19?",
            "answer": "The primary symptoms of COVID-19 include fever, cough, fatigue, and loss of taste or smell.",
            "expected": 5.0
        },
        {
            "name": "Minor Miss (Target 4.0)",
            "context": "The primary symptoms of COVID-19 include fever, cough, fatigue, and loss of taste or smell.",
            "question": "What are the symptoms of COVID-19?",
            "answer": "The symptoms of COVID-19 are fever, cough, and fatigue.",
            "expected": 4.0
        },
        {
            "name": "Major Miss / Vague (Target 3.0)",
            "context": "The primary symptoms of COVID-19 include fever, cough, fatigue, and loss of taste or smell.",
            "question": "What are the symptoms of COVID-19?",
            "answer": "COVID-19 causes fever and some other issues.",
            "expected": 3.0
        },
        {
            "name": "Hallucination (Control)",
            "context": "The primary symptoms of COVID-19 include fever, cough, fatigue, and loss of taste or smell.",
            "question": "What are the symptoms of COVID-19?",
            "answer": "COVID-19 symptoms include purple spots on the skin and hair loss.",
            "expected": 1.0
        }
    ]
    
    for case in test_cases:
        print(f"--- Testing Case: {case['name']} ---")
        print(f"Context: {case['context']}")
        print(f"Answer:  {case['answer']}")
        
        # Evaluate
        # Note: evaluate_rag_response expects a list of source dicts for context
        sources = [{'text': case['context']}]
        result = evaluate_rag_response(rag, case['question'], case['answer'], sources)
        
        score = result['Faithfulness']
        reasoning = result['Faithfulness_Reasoning']
        
        print(f"SCORE: {score}")
        print(f"REASONING: {reasoning}")
        print(f"Expected roughly: {case['expected']}")
        
        if abs(score - case['expected']) <= 0.5:
            print(">>> RESULT: MATCH")
        else:
            print(">>> RESULT: MISMATCH")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    run_partial_scoring_test()
