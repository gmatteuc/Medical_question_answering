"""
Medical QA - Generation Module
=============================================

This module provides functions for generating synthetic test queries
using the RAG model itself (LLM-as-a-Generator).
"""

import os
import random
from typing import List, Union, Optional

import numpy as np
import torch
import pandas as pd
import tqdm

# =============================================================================
# Generation Prompts
# =============================================================================

GEN_Q_PROMPT = """You are a medical expert creating a difficult exam. 
Your task is to generate a single, HIGHLY SPECIFIC question based *only* on the provided text.

Guidelines:
1. The answer to the question MUST be explicitly stated in the text. Do not ask for missing data.
2. If the text says "many" or "rarely", DO NOT ask for a specific percentage or number.
3. Focus on specific relationships, mechanisms, or conditions mentioned.
4. Avoid generic questions like "What is X?" or "What are the symptoms of X?" unless X is very rare.
5. Output ONLY the question. Do not provide any explanation or reasoning.

Text: {text}

Question:"""

VALIDATION_PROMPT = """You are an expert evaluator. 
Task: Determine if the following question can be answered PURELY based on the provided text.
Text: {text}
Question: {question}

Does the text contain the answer? Respond with only "YES" or "NO"."""

# =============================================================================
# Generation Functions
# =============================================================================

def validate_question(rag, text: str, question: str) -> bool:
    """
    Validates if the question is answerable from the text using the LLM.
    
    Args:
        rag: The RAG pipeline instance.
        text (str): The source text context.
        question (str): The generated question.
        
    Returns:
        bool: True if the question is answerable, False otherwise.
    """
    messages = [{"role": "user", "content": VALIDATION_PROMPT.format(text=text, question=question)}]
    
    inputs = rag.tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt", 
        return_dict=True
    ).to(rag.device)
    
    with torch.no_grad():
        outputs = rag.model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
    result = rag.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().upper()
    return "YES" in result

def generate_synthetic_queries(
    rag, 
    num_queries: int = 100, 
    output_path: str = '../output/generated_test_queries.csv', 
    seed: int = 42, 
    force: bool = False
) -> pd.DataFrame:
    """
    Generates synthetic test queries from the knowledge base.
    
    Args:
        rag (RAGPipeline): The initialized RAG pipeline.
        num_queries (int): Number of queries to generate.
        output_path (str): Path to save/load the queries (CSV preferred).
        seed (int): Random seed for reproducibility.
        force (bool): If True, regenerates queries even if file exists.
        
    Returns:
        pd.DataFrame: DataFrame containing 'question' and 'source_context'.
    """
    
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if os.path.exists(output_path) and not force:
        print(f"Loading existing queries from {output_path}...")
        if output_path.endswith('.csv'):
            df = pd.read_csv(output_path)
            print(f"Loaded {len(df)} queries.")
            return df
        else:
            # Legacy txt support
            with open(output_path, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
            return pd.DataFrame({'question': queries, 'source_context': [''] * len(queries)})

    # Sample chunks from the knowledge base
    print(f"Generating {num_queries} synthetic questions from the Knowledge Base...")
    
    # NEW LOGIC: Group by original_id to ensure full context
    if 'original_id' in rag.df.columns and 'full_answer' in rag.df.columns:
        print("Using document-level generation (grouping chunks by original_id)...")
        # Create a unique documents dataframe
        # We drop duplicates on original_id to get one row per document
        docs_df = rag.df[['original_id', 'full_answer']].drop_duplicates('original_id')
        
        # Filter for texts with sufficient length
        long_texts = docs_df[docs_df['full_answer'].str.len() > 500]
        if len(long_texts) == 0:
            print("Warning: No documents > 500 chars found. Using all documents.")
            long_texts = docs_df
            
        use_full_doc = True
    else:
        print("Warning: 'original_id' or 'full_answer' not found. Falling back to chunk-based generation.")
        # Fallback to existing logic
        long_texts = rag.df[rag.df['text'].str.len() > 500]
        if len(long_texts) == 0:
            print("Warning: No texts > 500 chars found. Using all texts.")
            long_texts = rag.df
        use_full_doc = False
        
    generated_data = []
    attempts = 0
    max_attempts = num_queries * 5 # Allow for retries
    
    pbar = tqdm.tqdm(total=num_queries, desc="Gen Valid Queries", ascii=True, ncols=80)
    
    while len(generated_data) < num_queries and attempts < max_attempts:
        attempts += 1
        
        # Sample one random text
        row = long_texts.sample(1).iloc[0]
        
        if use_full_doc:
            # Use the full answer as context
            full_text = row['full_answer']
            # Truncate if excessively long (e.g. > 6000 chars) to avoid context window issues for generation, 
            # but keep it large enough to be "full" context.
            text_snippet = full_text[:6000] 
            source_context = full_text # Store the FULL text for the Judge
        else:
            # Chunk based
            text_snippet = row['text'][:1000]
            source_context = text_snippet
        
        # Generate Question
        messages = [{"role": "user", "content": GEN_Q_PROMPT.format(text=text_snippet)}]
        
        inputs = rag.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt", 
            return_dict=True
        ).to(rag.device)
        
        with torch.no_grad():
            outputs = rag.model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
            
        q = rag.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        # Post-processing
        for marker in ["Explanation:", "**Explanation:**", "Answer:", "**Answer:**"]:
            if marker in q:
                q = q.split(marker)[0].strip()
        
        # Validate
        if validate_question(rag, text_snippet, q):
            generated_data.append({'question': q, 'source_context': source_context})
            pbar.update(1)
            
    pbar.close()

    # Add a few out-of-domain manually
    ood_queries = [
        "What is the capital of France?",
        "Who won the 2024 Super Bowl?",
        "Who was Asclepius?",
        "who are you?",
        "How do I bake a chocolate cake?"
    ]
    for q in ood_queries:
        generated_data.append({'question': q, 'source_context': 'OUT_OF_DOMAIN'})

    # Create DataFrame
    df_queries = pd.DataFrame(generated_data)

    # Save generated queries
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # If path ends in .txt, save as txt (legacy), else csv
    if output_path.endswith('.txt'):
        with open(output_path, 'w', encoding='utf-8') as f:
            for q in df_queries['question']:
                f.write(q + "\n")
    else:
        df_queries.to_csv(output_path, index=False)

    # Save detailed report for inspection (similar to test script output)
    report_path = output_path.replace('.csv', '_detailed.txt').replace('.txt', '_detailed.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        for i, row in df_queries.iterrows():
            f.write(f"=== QUERY {i+1} ===\n")
            f.write(f"Context Snippet:\n{str(row['source_context'])[:500]}...\n\n")
            f.write(f"Generated Question:\n{row['question']}\n\n")
            f.write(f"Validation: YES (Automatically Filtered)\n")
            f.write("="*50 + "\n\n")

    print(f"Generated {len(df_queries)} queries (including OOD).")
    print(f"Queries saved to {output_path}")
    print(f"Detailed report saved to {report_path}")
    
    return df_queries
