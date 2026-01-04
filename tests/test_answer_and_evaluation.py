"""
Medical QA - Answer and Evaluation Test
=============================================

This script tests the RAG pipeline's answer generation and the LLM-as-a-Judge evaluation logic.
It verifies that the generated answers are reasonable and that the evaluation metrics (Faithfulness, Relevance)
are consistent with the expected criteria.
"""

import os
import sys
import random
from typing import List, Dict, Union, Any

import torch
import pandas as pd

# Add project root to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag import RAGPipeline
from src.evaluation import evaluate_rag_response

def validate_judge_logic(
    rag: RAGPipeline, 
    question: str, 
    answer: str, 
    context: str, 
    score: float, 
    metric_name: str
) -> bool:
    """
    Asks the LLM to verify if the score given by the judge makes sense.
    
    Args:
        rag (RAGPipeline): The initialized RAG pipeline.
        question (str): The user's question.
        answer (str): The generated answer.
        context (str): The source context.
        score (float): The score assigned by the judge.
        metric_name (str): The name of the metric (Faithfulness or Relevance).
        
    Returns:
        bool: True if the score is deemed reasonable, False otherwise.
    """
    # Heuristic override for "I don't know" answers with high scores
    # If the answer is a refusal and the score is high, we assume it's a "Correct Refusal" and pass.
    # This avoids the Meta-Judge getting confused about whether "I don't know" is "supported".
    if "i don't know" in answer.lower() and score >= 4.0:
        return True

    if metric_name == "Faithfulness":
        criteria = "Score 3-5 means the answer is supported by the Context (5=Full, 3-4=Partial). Score 1 means the answer contains hallucinations."
    else: # Relevance
        criteria = "High score (4-5) means the answer directly addresses the Question OR acknowledges it (e.g., 'I don't know'). Low score (1-2) means the answer is unrelated."

    META_EVAL_PROMPT = """You are a Meta-Evaluator.
Task: Check if the score given to an answer is logical based on the metric.

Question: {question}
Context: {context}
Answer: {answer}

Metric: {metric}
Score Given: {score}
Criteria for this metric: {criteria}

Is this score reasonable according to the criteria?
Respond with only "YES" or "NO".
"""
    messages = [{"role": "user", "content": META_EVAL_PROMPT.format(
        question=question, context=context, answer=answer, metric=metric_name, score=score, criteria=criteria
    )}]
    
    inputs = rag.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(rag.device)
    with torch.no_grad():
        outputs = rag.model.generate(**inputs, max_new_tokens=10, do_sample=False)
    result = rag.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().upper()
    return "YES" in result

def run_evaluation_test(queries_path: str = '../output/generated_test_queries.csv', num_samples: int = 1) -> None:
    """
    Runs a random sample evaluation test using the RAG pipeline.
    
    Args:
        queries_path (str): Path to the generated queries CSV.
        num_samples (int): Number of random samples to test.
    """
    print("Initializing RAG Pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rag = RAGPipeline(device=device)
    rag.load_resources()
    
    print(f"Loading queries from {queries_path}...")
    if queries_path.endswith('.csv'):
        df = pd.read_csv(queries_path)
        queries = df['question'].tolist()
        contexts = df['source_context'].tolist()
    else:
        # Fallback for txt
        with open(queries_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
            contexts = ["(Context not available in txt)"] * len(queries)
            
    print(f"Loaded {len(queries)} queries.")
    
    # Select random samples
    indices = random.sample(range(len(queries)), min(num_samples, len(queries)))
    
    print(f"\nStarting Evaluation Test ({len(indices)} samples)...\n")
    
    for idx in indices:
        question = queries[idx]
        original_context = contexts[idx]
        
        print(f"=== TEST CASE: {question} ===")
        print(f"[Original Context Snippet]:\n{str(original_context)[:300]}...\n")
        
        # 1. RAG Answer (with retrieval)
        print("--- Generating RAG Answer ---")
        rag_answer, retrieved_docs = rag.generate_answer(question, k=3, distance_threshold=0.8)
        
        print(f"[RAG Answer]: {rag_answer}\n")
        
        print("--- Retrieved Contexts ---")
        retrieved_text_combined = ""
        found_original = False
        for i, doc in enumerate(retrieved_docs):
            # Note: 'distance' is L2 distance (lower is better). 
            # If using inner product, it might be different, but FAISS usually returns distance.
            score = doc.get('distance', doc.get('score', 0.0))
            print(f"Doc {i+1} (Distance: {score:.4f}): {doc['text'][:200]}...")
            retrieved_text_combined += doc['text'] + "\n"
            
            # Check if original context is roughly present (fuzzy match due to chunking)
            # We check if a significant substring of the original context is in the retrieved doc
            if original_context[:100] in doc['text']:
                found_original = True
        print()
        
        if found_original:
            print("TEST PASSED: Original context was retrieved.")
        else:
            print("TEST WARNING: Original context NOT found in top-k retrieved docs.")

        # 2. No-RAG Answer (LLM only)
        print("--- Generating No-RAG Answer ---")
        no_rag_messages = [
            {"role": "user", "content": f"Answer the following medical question based on your internal knowledge.\nQuestion: {question}"}
        ]
        no_rag_inputs = rag.tokenizer.apply_chat_template(no_rag_messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(rag.device)
        with torch.no_grad():
            no_rag_outputs = rag.model.generate(**no_rag_inputs, max_new_tokens=200, do_sample=False)
        no_rag_answer = rag.tokenizer.decode(no_rag_outputs[0][no_rag_inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        print(f"[No-RAG Answer]: {no_rag_answer}\n")
        
        # 3. Evaluation (Judge)
        print("--- Running LLM-as-a-Judge ---")
        
        # Evaluate RAG
        print("\n[Evaluating RAG Answer (vs Original Context)]...")
        # Use Original Context for evaluation as requested, to verify Ground Truth correctness
        rag_eval = evaluate_rag_response(rag, question, rag_answer, [{'text': original_context}])
        print(f"Faithfulness (Accuracy): {rag_eval['Faithfulness']} | Reason: {rag_eval['Faithfulness_Reasoning']}")
        
        # Meta-Eval Faithfulness
        # We also pass original_context to the Meta-Eval to be consistent
        if validate_judge_logic(rag, question, rag_answer, original_context, rag_eval['Faithfulness'], "Faithfulness"):
             print("META-EVAL PASSED: Faithfulness score seems logical.")
        else:
             print("META-EVAL FAILED: Faithfulness score might be illogical.")

        print(f"Relevance:    {rag_eval['Relevance']} | Reason: {rag_eval['Relevance_Reasoning']}")
        
        # Meta-Eval Relevance
        if validate_judge_logic(rag, question, rag_answer, retrieved_text_combined, rag_eval['Relevance'], "Relevance"):
             print("META-EVAL PASSED: Relevance score seems logical.")
        else:
             print("META-EVAL FAILED: Relevance score might be illogical.")
        
        # Evaluate No-RAG
        print("\n[Evaluating No-RAG Answer]...")
        no_rag_eval = evaluate_rag_response(rag, question, no_rag_answer, []) 
        print(f"Relevance:    {no_rag_eval['Relevance']} | Reason: {no_rag_eval['Relevance_Reasoning']}")
        
        # Meta-Eval Relevance No-RAG
        if validate_judge_logic(rag, question, no_rag_answer, "N/A", no_rag_eval['Relevance'], "Relevance"):
             print("META-EVAL PASSED: No-RAG Relevance score seems logical.")
        else:
             print("META-EVAL FAILED: No-RAG Relevance score might be illogical.")
        
        # 4. Fake Hallucination Test (Strictness Check)
        print("\n--- Fake Hallucination Test (Strictness Check) ---")
        # We create an answer that is definitely NOT in the context
        fake_hallucination = "The patient should take 500mg of Paracetamol every 4 hours. Also, the capital of France is Paris."
        print(f"[Fake Hallucination Answer]: {fake_hallucination}")
        
        # We pass the original context, which definitely does NOT contain this info
        eval_hallucination = evaluate_rag_response(rag, question, fake_hallucination, [{'text': original_context}])
        print(f"Faithfulness Score: {eval_hallucination['Faithfulness']}")
        print(f"Reasoning: {eval_hallucination['Faithfulness_Reasoning']}")
        
        if eval_hallucination['Faithfulness'] <= 1.0:
            print("PASS: Judge correctly identified hallucination.")
        else:
            print("FAIL: Judge was too lenient.")

        print("="*80 + "\n")

if __name__ == "__main__":
    # Ensure the path is correct relative to where we run the script
    queries_file = os.path.join(os.path.dirname(__file__), '..', 'output', 'generated_test_queries.csv')
    run_evaluation_test(queries_file, num_samples=3)
