"""
Medical QA - Out-of-Domain Test (Chocolate Cake)
=============================================

This script tests the RAG pipeline's ability to handle out-of-domain queries.
It specifically checks if the system correctly abstains from answering irrelevant questions
(like "How do I bake a chocolate cake?") when configured with safe prompts and thresholds.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag import RAGPipeline
from src.evaluation import evaluate_rag_response

def test_chocolate_cake() -> None:
    """
    Runs the chocolate cake test case.
    Compares Robust RAG (Abstain) vs Naive RAG (Hallucination).
    """
    rag = RAGPipeline()
    rag.load_resources()
    
    query = "How do I bake a chocolate cake?"
    
    print(f"\nQuery: {query}")
    
    # 1. Robust RAG (Abstain + Safe Prompt)
    print("\n--- Scenario 1: Robust RAG (Abstain + Safe Prompt) ---")
    ans_robust, docs_robust = rag.generate_answer(query, distance_threshold=0.8, safe_prompt=True)
    print(f"Answer: {ans_robust}")
    
    # Define Ground Truth Context (as in the main evaluation pipeline)
    ground_truth_context = [{'text': 'OUT_OF_DOMAIN'}]

    # Evaluate Robust
    print("Evaluating Robust RAG (against Ground Truth: OUT_OF_DOMAIN)...")
    eval_robust = evaluate_rag_response(rag, query, ans_robust, ground_truth_context)
    print(f"Faithfulness: {eval_robust['Faithfulness']}")
    print(f"Reasoning: {eval_robust['Faithfulness_Reasoning']}")
    
    # 2. Naive RAG (No Abstain + Unsafe Prompt)
    print("\n--- Scenario 2: Naive RAG (No Abstain + Unsafe Prompt) ---")
    ans_naive, docs_naive = rag.generate_answer(query, distance_threshold=100.0, safe_prompt=False)
    print(f"Answer: {ans_naive}")
    
    # Evaluate Naive
    print("Evaluating Naive RAG (against Ground Truth: OUT_OF_DOMAIN)...")
    eval_naive = evaluate_rag_response(rag, query, ans_naive, ground_truth_context)
    print(f"Faithfulness: {eval_naive['Faithfulness']}")
    print(f"Reasoning: {eval_naive['Faithfulness_Reasoning']}")

    # Assertions
    print("\n--- Test Results ---")
    robust_score = float(eval_robust['Faithfulness'])
    naive_score = float(eval_naive['Faithfulness'])

    if robust_score > 1.0 and naive_score <= 1.0:
        print("TEST PASSED: Robust RAG abstained (High Faithfulness) and Naive RAG hallucinated (Low Faithfulness).")
    else:
        print(f"TEST FAILED: Robust Score={robust_score}, Naive Score={naive_score}")
        # Raise error to make CI/CD fail if needed
        if not (robust_score > 1.0 and naive_score <= 1.0):
            sys.exit(1)

if __name__ == "__main__":
    test_chocolate_cake()
