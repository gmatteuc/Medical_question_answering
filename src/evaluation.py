"""
Medical QA - Evaluation Module
=============================================

This module provides functions for the LLM-as-a-Judge evaluation pipeline.
It includes prompts and scoring logic for Faithfulness and Relevance.
"""

import os
import gc
import re
import textwrap
from typing import List, Dict, Tuple, Optional, Union

import torch
import pandas as pd
import tqdm
from scipy import stats
from rouge_score import rouge_scorer

# =============================================================================
# Evaluation Prompts
# =============================================================================

# CHAIN-OF-THOUGHT PROMPT: Forces the model to verify specific claims and handle "I don't know" logic explicitly.
FAITHFULNESS_PROMPT = """You are a strict auditor for a RAG system.
Your ONLY job is to check if the Answer is derived *exclusively* from the provided Context.
You must IGNORE your internal knowledge. If the answer contains facts not in the context, it is a HALLUCINATION, even if those facts are true in the real world.

<Question>
{question}
</Question>

<Context>
{context}
</Context>

<Answer>
{answer}
</Answer>

Scoring Rules:
1. Score 5.0 (Fully Faithful): The Answer contains ONLY information present in the Context.
2. Score 4.0 (Faithful): The Answer contains ONLY information from the Context, but misses minor details.
3. Score 3.0 (Partially Faithful): The Answer contains ONLY information from the Context, but is vague.
4. Score 2.0 (Incomplete): The Answer is very vague or incomplete, but still grounded in Context.
5. Score 1.0 (Hallucination): The Answer contains ANY specific information (names, numbers, causes, treatments) that is NOT in the Context.

CRITICAL INSTRUCTIONS:
- If the Answer mentions a specific drug, gene, percentage, or symptom NOT in the Context -> Score 1.0.
- If the Answer is "I don't know" and the Context does not have the answer -> Score 5.0.
- Do NOT judge based on whether the answer is "correct" medically. Judge ONLY if it is in the Context.

Output Format:
Reasoning: <Step 1: List facts in Answer. Step 2: Check if each fact is in Context. Step 3: Assign Score>
Score: <number>"""

RELEVANCE_PROMPT = """You are an expert evaluator.
Your task is to check if the Answer addresses the User's Question.

Question:
{question}

Answer:
{answer}

Scoring Rules:
1. Score 5.0 (Fully Relevant): The Answer directly answers the user's question with specific information.
2. Score 4.0 (Relevant): The Answer addresses the main question but misses a secondary part.
3. Score 3.0 (Partially Relevant): The Answer addresses the general topic but misses the specific question.
4. Score 2.0 (Partially Relevant): The Answer addresses the main question is vague and misses the specific question.
5. Score 1.0 (Irrelevant / Refusal): The Answer talks about something completely different OR refuses to answer (e.g., "I don't know", "I cannot provide", "Consult a doctor") when the user asked for specific facts.

Important:
- If the user asks for a number/fact and the model says "I don't know", this is Score 1.0 (it failed to be relevant to the user's need).

Output Format:
Reasoning: <Explain why it fits the score>
Score: <number>"""

# =============================================================================
# Evaluation Functions
# =============================================================================

def call_llm_judge(rag, prompt: str) -> Tuple[float, str]:
    """
    Uses the RAG pipeline's loaded LLM to generate a score based on the prompt.
    Returns a tuple: (score, reasoning_text)
    """
    messages = [{"role": "user", "content": prompt}]
    
    inputs = rag.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(rag.device)
    
    with torch.no_grad():
        outputs = rag.model.generate(
            **inputs, 
            max_new_tokens=512, # Increased to allow for full reasoning
            do_sample=False,
            temperature=0.0
        )
    
    generated_text = rag.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    
    # Robust parsing for "Score: <number>"
    score = 0.0
    try:
        # Look for "Score:" followed by a number (int or float)
        match = re.search(r"Score:\s*([0-9]*\.?[0-9]+)", generated_text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            if 0 <= val <= 5:
                score = val
        else:
            # Fallback: look for any number between 0 and 5 at the end of the string
            numbers = re.findall(r"([0-9]*\.?[0-9]+)", generated_text)
            if numbers:
                for num_str in reversed(numbers):
                    try:
                        val = float(num_str)
                        if 0 <= val <= 5:
                            score = val
                            break
                    except:
                        continue
    except:
        score = 0.0
        
    return score, generated_text

def evaluate_rag_response(rag, query, answer, sources):
    """
    Evaluates a single RAG response for Faithfulness and Relevance.
    Returns scores and reasoning.
    """
    # Prepare context string
    context_text = "\n".join([f"- {s['text']}" for s in sources])
    
    # Pre-check for Refusals (I don't know)
    # If the model refuses to answer, it is technically "Faithful" (it didn't hallucinate).
    # We handle this programmatically to avoid LLM confusion.
    clean_answer = answer.strip().lower()
    if clean_answer.startswith("i don't know") or clean_answer.startswith("i do not know"):
        faith_score = 5.0
        faith_reasoning = "Reasoning: The answer is a refusal ('I don't know'). This is considered fully faithful as the model avoided hallucinating information not in the context.\nScore: 5.0"
    else:
        # 1. Evaluate Faithfulness via LLM
        faith_prompt = FAITHFULNESS_PROMPT.format(context=context_text, answer=answer, question=query)
        faith_score, faith_reasoning = call_llm_judge(rag, faith_prompt)
    
    # 2. Evaluate Relevance
    rel_prompt = RELEVANCE_PROMPT.format(question=query, answer=answer)
    rel_score, rel_reasoning = call_llm_judge(rag, rel_prompt)
    
    return {
        "Faithfulness": faith_score,
        "Faithfulness_Reasoning": faith_reasoning,
        "Relevance": rel_score,
        "Relevance_Reasoning": rel_reasoning
    }

def run_statistical_tests(plot_df, metric_name="Faithfulness"):
    """
    Runs Wilcoxon Signed-Rank Test for the given metric between scenarios.
    """
    print(f"\n=== Statistical Significance Tests (Wilcoxon Signed-Rank) ===")
    print(f"Comparing Median {metric_name} between scenarios (Paired Data)\n")

    # Pivot for easy pairing
    # Use groupby and unstack to handle potential duplicates safely
    pivot_data = plot_df.groupby(['Query', 'Scenario'])[metric_name].mean().unstack()

    pairs_to_test = [
        ("RAG + Abstain", "No RAG (LLM Only)"),
        ("RAG (No Abstain)", "No RAG (LLM Only)"),
        ("RAG + Abstain", "RAG (No Abstain)")
    ]

    for s1, s2 in pairs_to_test:
        # Get paired data, dropping queries where either is missing (e.g. abstained)
        if s1 not in pivot_data.columns or s2 not in pivot_data.columns:
            continue
            
        paired_data = pivot_data[[s1, s2]].dropna()
        
        if len(paired_data) < 5:
            print(f"{s1} vs {s2}: Not enough data points ({len(paired_data)})")
            continue
            
        # Wilcoxon test
        try:
            stat, p_val = stats.wilcoxon(paired_data[s1], paired_data[s2])
            
            significance = ""
            if p_val < 0.001: significance = "***"
            elif p_val < 0.01: significance = "**"
            elif p_val < 0.05: significance = "*"
            else: significance = "(ns)"
            
            print(f"{s1} vs {s2}:")
            print(f"   n={len(paired_data)}, Statistic={stat:.1f}, p-value={p_val:.4f} {significance}")
            print(f"   Median Diff: {paired_data[s1].median() - paired_data[s2].median():.2f}")
        except ValueError as e:
            print(f"{s1} vs {s2}: Error (likely all differences are zero) - {e}")

def get_judge_explanation(rag, prompt):
    """
    Runs the judge and returns the full generated text (Reasoning + Score).
    """
    messages = [{"role": "user", "content": prompt}]
    inputs = rag.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(rag.device)
    with torch.no_grad():
        outputs = rag.model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return rag.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

def analyze_smoking_guns(results_df, rag=None, context_map=None, output_file=None):
    """
    Identifies and prints 'Smoking Gun' examples where RAG is faithful and No-RAG hallucinates.
    Uses stored reasoning if available, otherwise re-runs judge if rag is provided.
    
    Args:
        results_df: DataFrame containing evaluation results.
        rag: Optional RAG pipeline for fallback execution.
        context_map: Optional dictionary mapping Query -> Original Context text.
        output_file: Optional path to save the report.
    """
    # Helper to log to both console and file
    def log(text):
        print(text)
        if output_file:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(text + "\n")

    # Clear file if it exists
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("")

    # Pivot to compare RAG vs No RAG
    # Check which RAG scenario is available
    available_scenarios = results_df['Scenario'].unique()
    if "RAG (No Abstain)" in available_scenarios:
        rag_scenario = "RAG (No Abstain)"
    elif "RAG + Abstain" in available_scenarios:
        rag_scenario = "RAG + Abstain"
    else:
        log("Error: No RAG scenario found for comparison.")
        return

    norag_scenario = "No RAG (LLM Only)"

    rag_df = results_df[results_df['Scenario'] == rag_scenario].set_index('Query')
    norag_df = results_df[results_df['Scenario'] == norag_scenario].set_index('Query')

    # Combine into a comparison dataframe
    comparison = pd.DataFrame({
        'RAG_Answer': rag_df['Answer'],
        'RAG_Faithfulness': rag_df['Faithfulness'],
        'RAG_Relevance': rag_df['Relevance'],
        'RAG_Reasoning': rag_df.get('Faithfulness_Reasoning', pd.Series([None]*len(rag_df), index=rag_df.index)),
        'RAG_Relevance_Reasoning': rag_df.get('Relevance_Reasoning', pd.Series([None]*len(rag_df), index=rag_df.index)),
        'RAG_Meta_Faithfulness': rag_df.get('Meta_Faithfulness_Passed', pd.Series([None]*len(rag_df), index=rag_df.index)),
        'RAG_Meta_Relevance': rag_df.get('Meta_Relevance_Passed', pd.Series([None]*len(rag_df), index=rag_df.index)),
        
        'NoRAG_Answer': norag_df['Answer'],
        'NoRAG_Faithfulness': norag_df['Faithfulness'],
        'NoRAG_Relevance': norag_df['Relevance'],
        'NoRAG_Reasoning': norag_df.get('Faithfulness_Reasoning', pd.Series([None]*len(norag_df), index=norag_df.index)),
        'NoRAG_Relevance_Reasoning': norag_df.get('Relevance_Reasoning', pd.Series([None]*len(norag_df), index=norag_df.index)),
        'NoRAG_Meta_Relevance': norag_df.get('Meta_Relevance_Passed', pd.Series([None]*len(norag_df), index=norag_df.index))
    }).dropna(subset=['RAG_Answer', 'NoRAG_Answer'])

    # Calculate Gap
    comparison['Gap'] = comparison['RAG_Faithfulness'] - comparison['NoRAG_Faithfulness']

    # Filter for "Smoking Guns": RAG is Faithful (>= 4.0) AND No-RAG Hallucinates (<= 2.0)
    smoking_guns = comparison[
        (comparison['RAG_Faithfulness'] >= 4.0) & 
        (comparison['NoRAG_Faithfulness'] <= 2.0)
    ].sort_values(by='Gap', ascending=False)

    # Filter for "Significant Improvements": RAG is Faithful (>= 4.0) AND No-RAG is Vague/Partial (<= 3.0)
    # Exclude the smoking guns to avoid duplication
    significant_improvements = comparison[
        (comparison['RAG_Faithfulness'] >= 4.0) & 
        (comparison['NoRAG_Faithfulness'] <= 3.0) &
        (comparison['NoRAG_Faithfulness'] > 2.0)
    ].sort_values(by='Gap', ascending=False)

    log(f"\n=== SMOKING GUN ANALYSIS ===")
    log(f"Total Comparisons: {len(comparison)}")
    log(f"Smoking Guns Found: {len(smoking_guns)} (Cases where RAG was correct and No-RAG hallucinated)")
    log(f"Significant Improvements: {len(significant_improvements)} (Cases where RAG was correct and No-RAG was vague/partial)")
    log("="*80)

    # Display Top Examples (All of them if saving to file, otherwise top 3)
    limit = len(smoking_guns) if output_file else 3
    
    def print_example(i, query, row, label="EXAMPLE"):
        # Clean query display (remove artifacts like "**Explanation:**")
        clean_query = query.split("**Explanation:**")[0].strip()
        
        log(f"\n{label} {i+1}:")
        log(f"QUESTION: {clean_query}")
        
        # Display Original Context if available
        if context_map and clean_query in context_map:
            log(f"\n[Original Context Snippet]:\n{str(context_map[clean_query])[:300]}...")
        
        log("-" * 40)
        
        # Get reasoning (from DF or re-run)
        rag_reasoning = row['RAG_Reasoning']
        rag_rel_reasoning = row['RAG_Relevance_Reasoning']
        norag_reasoning = row['NoRAG_Reasoning']
        norag_relevance_reasoning = row.get('NoRAG_Relevance_Reasoning', None)
        
        # If reasoning is missing/NaN and we have the rag pipeline, try to re-run (fallback)
        if rag:
            try:
                retrieved_docs = rag.retrieve(clean_query, k=5)
                context_text = "\n".join([f"- {s['text']}" for s in retrieved_docs])
                
                if pd.isna(rag_reasoning):
                    rag_prompt = FAITHFULNESS_PROMPT.format(context=context_text, answer=row['RAG_Answer'], question=clean_query)
                    _, rag_reasoning = call_llm_judge(rag, rag_prompt)
                
                # For No-RAG, if we have context_map, we can evaluate Faithfulness (Accuracy) against Ground Truth
                if (pd.isna(norag_reasoning) or "N/A" in str(norag_reasoning)) and context_map and clean_query in context_map:
                     original_context = context_map[clean_query]
                     norag_prompt = FAITHFULNESS_PROMPT.format(context=original_context, answer=row['NoRAG_Answer'], question=clean_query)
                     norag_score, norag_reasoning = call_llm_judge(rag, norag_prompt)
                     # Update score for display
                     row['NoRAG_Faithfulness'] = norag_score

            except Exception as e:
                log(f"(Could not retrieve reasoning: {e})")

        log(f"--- RAG SYSTEM (With Context) ---")
        log(f"Faithfulness Score: {row['RAG_Faithfulness']:.1f} / 5.0")
        log(f"Relevance Score:    {row['RAG_Relevance']:.1f} / 5.0")
        log(f"Meta-Eval: Faithfulness={'PASS' if row['RAG_Meta_Faithfulness'] else 'FAIL'}, Relevance={'PASS' if row['RAG_Meta_Relevance'] else 'FAIL'}")
        log(f"Generated Answer:\n{row['RAG_Answer']}")
        
        if rag_reasoning:
            log(f"\n[JUDGE REASONING (Faithfulness)]:\n{rag_reasoning}")
        if rag_rel_reasoning:
            log(f"\n[JUDGE REASONING (Relevance)]:\n{rag_rel_reasoning}")
            
        log("-" * 40)
        
        log(f"--- NO-RAG SYSTEM (LLM Only) ---")
        log(f"Faithfulness Score: {row['NoRAG_Faithfulness']:.1f} / 5.0")
        log(f"Relevance Score:    {row['NoRAG_Relevance']:.1f} / 5.0")
        log(f"Meta-Eval: Relevance={'PASS' if row['NoRAG_Meta_Relevance'] else 'FAIL'}")
        log(f"Generated Answer (Likely Hallucination/External Info):\n{row['NoRAG_Answer']}")
        
        if norag_reasoning and not pd.isna(norag_reasoning):
             log(f"\n[JUDGE REASONING (Faithfulness)]:\n{norag_reasoning}")
        if norag_relevance_reasoning and not pd.isna(norag_relevance_reasoning):
             log(f"\n[JUDGE REASONING (Relevance)]:\n{norag_relevance_reasoning}")
        
        if (not norag_reasoning or pd.isna(norag_reasoning)) and (not norag_relevance_reasoning or pd.isna(norag_relevance_reasoning)):
             log(f"\n[JUDGE REASONING]:\nN/A (No Context Provided)")
             
        log("="*80)

    # Print Smoking Guns
    if not smoking_guns.empty:
        log("\n>>> CATEGORY 1: SEVERE HALLUCINATIONS (RAG Correct vs No-RAG Wrong) <<<")
        for i, (query, row) in enumerate(smoking_guns.head(limit).iterrows()):
            print_example(i, query, row, label="SMOKING GUN")

    # Print Significant Improvements
    if not significant_improvements.empty:
        log("\n>>> CATEGORY 2: SIGNIFICANT IMPROVEMENTS (RAG Detailed vs No-RAG Vague) <<<")
        limit_imp = len(significant_improvements) if output_file else 3
        for i, (query, row) in enumerate(significant_improvements.head(limit_imp).iterrows()):
            print_example(i, query, row, label="IMPROVEMENT")

    
    if output_file:
        print(f"Smoking Gun analysis saved to: {output_file}")

def analyze_rag_gap(results_df, rag=None):
    """
    Performs a gap analysis between RAG and No-RAG scenarios.
    Identifies wins, losses, and specific examples.
    """
    print("Inizio analisi comparativa (Gap Analysis)...", flush=True)

    # Merge results to compare side-by-side
    print("Filtraggio dati...", flush=True)
    rag_df = results_df[results_df['Scenario'] == 'RAG + Abstain'].copy()
    rag_df = rag_df.set_index('Query')
    
    no_rag_df = results_df[results_df['Scenario'] == 'No RAG (LLM Only)'].copy()
    no_rag_df = no_rag_df.set_index('Query')

    print(f"Confronto {len(rag_df)} risposte RAG con {len(no_rag_df)} risposte No-RAG.", flush=True)

    # Find common queries
    common_queries = rag_df.index.intersection(no_rag_df.index)

    # Calculate gaps
    gaps = []
    for q in common_queries:
        rag_faith = rag_df.loc[q, 'Faithfulness']
        no_rag_faith = no_rag_df.loc[q, 'Faithfulness']
        
        # Skip if RAG abstained (None)
        if pd.isna(rag_faith):
            continue
            
        # Treat No-RAG None as 0.0 (though it shouldn't happen for No-RAG)
        if pd.isna(no_rag_faith):
            no_rag_faith = 0.0
            
        gap = rag_faith - no_rag_faith
        gaps.append({
            'Query': q,
            'RAG_Faithfulness': rag_faith,
            'NoRAG_Faithfulness': no_rag_faith,
            'Gap': gap,
            'RAG_Answer': rag_df.loc[q, 'Answer'],
            'NoRAG_Answer': no_rag_df.loc[q, 'Answer']
        })

    # Create DataFrame
    gap_df = pd.DataFrame(gaps)

    if not gap_df.empty:
        print(f"Trovate {len(gap_df)} comparazioni valide.", flush=True)
        
        # --- Win/Tie/Loss Stats ---
        wins = len(gap_df[gap_df['Gap'] > 0.1])
        ties = len(gap_df[(gap_df['Gap'] >= -0.1) & (gap_df['Gap'] <= 0.1)])
        losses = len(gap_df[gap_df['Gap'] < -0.1])
        
        print(f"\n--- Summary Stats ---")
        print(f"RAG Wins (Gap > 0.1):  {wins} ({wins/len(gap_df)*100:.1f}%)")
        print(f"Ties     (Gap +/-0.1): {ties} ({ties/len(gap_df)*100:.1f}%)")
        print(f"RAG Loses (Gap < -0.1): {losses} ({losses/len(gap_df)*100:.1f}%)")

        # --- RAG Wins (General) ---
        print("\n=== Top RAG Wins (Largest Positive Gap) ===\n", flush=True)
        top_wins = gap_df.sort_values(by='Gap', ascending=False).head(3)
        for _, row in top_wins.iterrows():
            print(f"Query: {row['Query']}")
            print(f"Gap: +{row['Gap']:.1f} (RAG: {row['RAG_Faithfulness']} vs No RAG: {row['NoRAG_Faithfulness']})")
            print("-" * 30)

        # --- RAG Losses ---
        print("\n=== RAG Losses: Where No-RAG was 'better' ===\n", flush=True)
        top_losses = gap_df.sort_values(by='Gap', ascending=True).head(3)
        
        if top_losses.iloc[0]['Gap'] < 0:
            for _, row in top_losses.iterrows():
                if row['Gap'] >= 0: continue # Stop if we reach positive gaps
                print(f"\n{'!'*20} LOSS ANALYSIS {'!'*20}")
                print(f"QUERY: {row['Query']}")
                print(f"Gap: {row['Gap']:.1f} (RAG: {row['RAG_Faithfulness']} vs No RAG: {row['NoRAG_Faithfulness']})")
                
                if rag:
                    print(f"GROUND TRUTH (Retrieved Context):")
                    try:
                        docs = rag.retrieve(row['Query'], k=5)
                        for i, d in enumerate(docs):
                            print(f"--- Doc {i+1} ---")
                            print(d['text'].strip())
                    except Exception as e:
                        print(f"(Error retrieving context: {e})")

                print(f"\n--- RAG Answer ---")
                print(row['RAG_Answer'])
                print(f"\n--- No RAG Answer ---")
                print(row['NoRAG_Answer'])
                print("-" * 50)
        else:
            print("Nessuna perdita significativa trovata!")

    else:
        print("Nessuna comparazione valida trovata (forse troppe astensioni?).", flush=True)

def validate_judge_logic(rag, question, answer, context, score, metric_name):
    """
    Asks the LLM to verify if the score given by the judge makes sense.
    """
    # Heuristic override for "I don't know" answers with high scores
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

def run_comprehensive_evaluation(rag, queries, contexts, detailed_output_path, results_path=None):
    """
    Runs the full evaluation pipeline on a list of queries.
    
    Args:
        rag: The RAGPipeline object.
        queries: List of question strings.
        contexts: List of ground truth context strings.
        detailed_output_path: Path to save the detailed text report.
        results_path: Optional path to save the results CSV.
        
    Returns:
        pd.DataFrame: The evaluation results.
    """
    results = []
    
    print(f"Starting Detailed Evaluation on {len(queries)} queries...")
    print(f"Detailed report will be saved to: {detailed_output_path}")

    with open(detailed_output_path, 'w', encoding='utf-8') as f_out:
        f_out.write("=== DETAILED EVALUATION REPORT ===\n\n")
        
        for i, (question, original_context) in enumerate(tqdm.tqdm(zip(queries, contexts), total=len(queries), desc="Evaluating")):
            
            # Periodic cleanup
            if i % 10 == 0:
                torch.cuda.empty_cache()
            
            f_out.write(f"=== TEST CASE {i+1}: {question} ===\n")
            f_out.write(f"[Original Context Snippet]:\n{str(original_context)[:300]}...\n\n")
            
            # --- 1. RAG Answer (With Abstain) ---
            # Threshold 0.8 means "Abstain if distance > 0.8"
            rag_answer_abstain, retrieved_docs = rag.generate_answer(question, k=3, distance_threshold=0.8, safe_prompt=True)
            
            # Check retrieval stats for logging
            retrieved_text_combined = ""
            for doc in retrieved_docs:
                retrieved_text_combined += doc['text'] + "\n"
                
            # --- 2. RAG Answer (No Abstain / Forced) ---
            # We force the model to answer by setting a huge threshold (100.0) AND removing the safety prompt
            if "i don't know" in rag_answer_abstain.lower():
                rag_answer_forced, _ = rag.generate_answer(question, k=3, distance_threshold=100.0, safe_prompt=False)
            else:
                rag_answer_forced = rag_answer_abstain

            f_out.write(f"--- RAG Answers ---\n")
            f_out.write(f"[Abstain Mode]: {rag_answer_abstain}\n")
            f_out.write(f"[Forced Mode]:  {rag_answer_forced}\n\n")
                
            # --- 3. No-RAG Answer ---
            no_rag_messages = [{"role": "user", "content": f"Answer the following medical question based on your internal knowledge.\nQuestion: {question}"}]
            no_rag_inputs = rag.tokenizer.apply_chat_template(no_rag_messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(rag.device)
            with torch.no_grad():
                no_rag_outputs = rag.model.generate(**no_rag_inputs, max_new_tokens=200, do_sample=False)
            no_rag_answer = rag.tokenizer.decode(no_rag_outputs[0][no_rag_inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            
            f_out.write(f"[No-RAG Answer]: {no_rag_answer}\n\n")
            
            # --- 4. Evaluation (Judge) ---
            f_out.write("--- Running LLM-as-a-Judge ---\n")
            
            # A. Evaluate RAG + Abstain
            eval_abstain = evaluate_rag_response(rag, question, rag_answer_abstain, [{'text': original_context}])
            
            f_out.write(f"\n[Scenario: RAG + Abstain]\n")
            f_out.write(f"Faithfulness: {eval_abstain['Faithfulness']}\nReasoning: {eval_abstain['Faithfulness_Reasoning']}\n")
            f_out.write(f"Relevance: {eval_abstain['Relevance']}\nReasoning: {eval_abstain['Relevance_Reasoning']}\n")

            meta_faith_abstain = validate_judge_logic(rag, question, rag_answer_abstain, original_context, eval_abstain['Faithfulness'], "Faithfulness")
            meta_rel_abstain = validate_judge_logic(rag, question, rag_answer_abstain, retrieved_text_combined, eval_abstain['Relevance'], "Relevance")
            
            results.append({
                "Scenario": "RAG + Abstain", 
                "Query": question,
                "Answer": rag_answer_abstain,
                "Abstained": "i don't know" in rag_answer_abstain.lower(),
                "Faithfulness": eval_abstain['Faithfulness'],
                "Faithfulness_Reasoning": eval_abstain['Faithfulness_Reasoning'],
                "Relevance": eval_abstain['Relevance'],
                "Relevance_Reasoning": eval_abstain['Relevance_Reasoning'],
                "Meta_Faithfulness_Passed": meta_faith_abstain,
                "Meta_Relevance_Passed": meta_rel_abstain
            })

            # B. Evaluate RAG (No Abstain)
            if rag_answer_forced != rag_answer_abstain:
                eval_forced = evaluate_rag_response(rag, question, rag_answer_forced, [{'text': original_context}])
                
                f_out.write(f"\n[Scenario: RAG (No Abstain)]\n")
                f_out.write(f"Faithfulness: {eval_forced['Faithfulness']}\nReasoning: {eval_forced['Faithfulness_Reasoning']}\n")
                f_out.write(f"Relevance: {eval_forced['Relevance']}\nReasoning: {eval_forced['Relevance_Reasoning']}\n")

                meta_faith_forced = validate_judge_logic(rag, question, rag_answer_forced, original_context, eval_forced['Faithfulness'], "Faithfulness")
                meta_rel_forced = validate_judge_logic(rag, question, rag_answer_forced, retrieved_text_combined, eval_forced['Relevance'], "Relevance")
            else:
                f_out.write(f"\n[Scenario: RAG (No Abstain)]\n(Same as Abstain)\n")
                eval_forced = eval_abstain
                meta_faith_forced = meta_faith_abstain
                meta_rel_forced = meta_rel_abstain

            results.append({
                "Scenario": "RAG (No Abstain)", 
                "Query": question,
                "Answer": rag_answer_forced,
                "Abstained": "i don't know" in rag_answer_forced.lower(),
                "Faithfulness": eval_forced['Faithfulness'],
                "Faithfulness_Reasoning": eval_forced['Faithfulness_Reasoning'],
                "Relevance": eval_forced['Relevance'],
                "Relevance_Reasoning": eval_forced['Relevance_Reasoning'],
                "Meta_Faithfulness_Passed": meta_faith_forced,
                "Meta_Relevance_Passed": meta_rel_forced
            })
            
            # C. Evaluate No-RAG
            # We evaluate against the Original Context to check for Hallucinations/Accuracy (Ground Truth Check)
            no_rag_eval = evaluate_rag_response(rag, question, no_rag_answer, [{'text': original_context}])
            
            f_out.write(f"\n[Scenario: No RAG]\n")
            f_out.write(f"Faithfulness: {no_rag_eval['Faithfulness']}\nReasoning: {no_rag_eval['Faithfulness_Reasoning']}\n")
            f_out.write(f"Relevance: {no_rag_eval['Relevance']}\nReasoning: {no_rag_eval['Relevance_Reasoning']}\n")

            meta_rel_norag = validate_judge_logic(rag, question, no_rag_answer, "N/A", no_rag_eval['Relevance'], "Relevance")
            meta_faith_norag = validate_judge_logic(rag, question, no_rag_answer, original_context, no_rag_eval['Faithfulness'], "Faithfulness")
            
            results.append({
                "Scenario": "No RAG (LLM Only)",
                "Query": question,
                "Answer": no_rag_answer,
                "Abstained": "i don't know" in no_rag_answer.lower(),
                "Faithfulness": no_rag_eval['Faithfulness'], 
                "Faithfulness_Reasoning": no_rag_eval['Faithfulness_Reasoning'],
                "Relevance": no_rag_eval['Relevance'],
                "Relevance_Reasoning": no_rag_eval['Relevance_Reasoning'],
                "Meta_Faithfulness_Passed": meta_faith_norag, 
                "Meta_Relevance_Passed": meta_rel_norag
            })
            
            f_out.write("="*80 + "\n\n")

    results_df = pd.DataFrame(results)
    if results_path:
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")
        
    return results_df

def generate_detailed_summary_report(results_df, output_file, context_map=None):
    """
    Generates a detailed text report summarizing the evaluation results side-by-side.
    Args:
        results_df: DataFrame containing evaluation results.
        output_file: Path to save the report.
        context_map: Optional dictionary mapping Query -> Original Context text.
    """
    # Ensure we don't have duplicates
    clean_results = results_df.drop_duplicates(subset=['Query', 'Scenario'])

    # Identify which columns are actually present
    potential_cols = ['Faithfulness', 'Relevance', 'Faithfulness_Reasoning', 'Relevance_Reasoning', 'Answer', 'Meta_Faithfulness_Passed', 'Meta_Relevance_Passed']
    cols_to_compare = [c for c in potential_cols if c in clean_results.columns]

    # Pivot to get scenarios side-by-side
    comparison = clean_results.pivot_table(index='Query', columns='Scenario', values=cols_to_compare, aggfunc='first')

    # Determine available scenarios
    all_possible_scenarios = ["RAG + Abstain", "RAG (No Abstain)", "No RAG (LLM Only)"]
    available_scenarios = [s for s in all_possible_scenarios if s in results_df['Scenario'].unique()]

    print(f"Generating summary analysis report to: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("================================================================================\n")
        f.write("DETAILED EVALUATION SUMMARY REPORT\n")
        f.write("================================================================================\n\n")
        
        for query in comparison.index:
            f.write(f"QUERY: {query}\n")
            
            # Display Original Context if available
            if context_map and query in context_map:
                context_snippet = str(context_map[query])[:300].replace("\n", " ") + "..."
                f.write(f"[Original Context Snippet]:\n{textwrap.fill(context_snippet, width=100, initial_indent='    ', subsequent_indent='    ')}\n")
            
            f.write("-" * 80 + "\n")
            
            for scenario in available_scenarios:
                f.write(f"[SCENARIO: {scenario}]\n")
                
                def get_val(col_name, scen):
                    if col_name in cols_to_compare:
                        try:
                            val = comparison.loc[query, (col_name, scen)]
                            if pd.isna(val): return "N/A"
                            return val
                        except KeyError:
                            return "N/A"
                    return "N/A (Column Missing)"

                faith = get_val('Faithfulness', scenario)
                rel = get_val('Relevance', scenario)
                ans = get_val('Answer', scenario)
                faith_r = get_val('Faithfulness_Reasoning', scenario)
                rel_r = get_val('Relevance_Reasoning', scenario)
                
                f.write(f"Faithfulness: {faith} | Relevance: {rel}\n")
                
                # Helper to clean and wrap text
                def clean_and_wrap(text, width=100, indent='    '):
                    if not isinstance(text, str): text = str(text)
                    # Replace literal \n with actual newlines for formatting
                    text = text.replace('\\n', '\n')
                    # Wrap preserving existing newlines
                    lines = text.split('\n')
                    wrapped_lines = [textwrap.fill(line, width=width, initial_indent=indent, subsequent_indent=indent) for line in lines]
                    return '\n'.join(wrapped_lines)

                f.write(f"Answer:\n{clean_and_wrap(ans)}\n")
                
                if faith_r and "N/A" not in str(faith_r):
                    f.write(f"Faithfulness Reasoning:\n{clean_and_wrap(faith_r)}\n")
                elif faith_r and "N/A" in str(faith_r) and scenario == "No RAG (LLM Only)":
                     # Special case: Show N/A reasoning for No-RAG Faithfulness if that's what we have
                     f.write(f"Faithfulness Reasoning:\n{clean_and_wrap(faith_r)}\n")
                
                if rel_r and "N/A" not in str(rel_r):
                    f.write(f"Relevance Reasoning:\n{clean_and_wrap(rel_r)}\n")
                
                f.write("\n")

            f.write("=" * 80 + "\n\n")

# =============================================================================
# New Analysis Functions (Refactored from Notebook)
# =============================================================================

def calculate_rouge_scores(results_df, context_map):
    """
    Calculates ROUGE-L scores for valid answers in the results dataframe.
    
    Args:
        results_df (pd.DataFrame): The evaluation results.
        context_map (dict): Mapping from Query to Context.
        
    Returns:
        pd.DataFrame: A subset of results_df with an added 'ROUGE-L' column.
    """
    if not context_map:
        print("WARNING: No context map provided. ROUGE analysis skipped.")
        return pd.DataFrame()

    # Map contexts to results
    df = results_df.copy()
    df['Context'] = df['Query'].map(context_map)

    # Filter: Exclude Abstentions and "I don't know"
    valid_mask = (
        (df['Abstained'] == False) & 
        (df['Answer'].notna()) & 
        (df['Context'].notna()) &
        (~df['Answer'].str.strip().str.lower().isin(["i don't know", "i don't know.", "i do not know."]))
    )
    valid_results = df[valid_mask].copy()
    
    print(f"Calculating ROUGE scores for {len(valid_results)} valid answers...")

    # Calculate ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    def get_rouge_l(row):
        try:
            # Score returns precision, recall, fmeasure. We use fmeasure.
            return scorer.score(row['Context'], row['Answer'])['rougeL'].fmeasure
        except Exception as e:
            return 0.0

    valid_results['ROUGE-L'] = valid_results.apply(get_rouge_l, axis=1)
    return valid_results

def calculate_evaluation_stats(results_df, total_queries):
    """
    Calculates comprehensive statistics for the evaluation results.
    
    Args:
        results_df (pd.DataFrame): The evaluation results.
        total_queries (int): Total number of queries evaluated.
        
    Returns:
        dict: A dictionary containing 'abstain_stats', 'score_stats', and 'risk_stats' dataframes.
    """
    # --- Abstention Statistics ---
    abstain_stats = results_df.groupby("Scenario")['Abstained'].sum().reset_index()
    abstain_stats['Total Queries'] = total_queries
    abstain_stats['Abstention Rate'] = (abstain_stats['Abstained'] / total_queries) * 100
    
    # --- Score Statistics ---
    score_stats = results_df.groupby("Scenario")[['Faithfulness', 'Relevance']].agg(['mean', 'median', 'std', 'count'])
    
    # --- Safety Analysis ---
    risk_stats = []
    present_scenarios = results_df['Scenario'].unique()
    
    for scenario in present_scenarios:
        scenario_data = results_df[results_df['Scenario'] == scenario]
        n_total = len(scenario_data)
        # Count hallucinations (Low Faithfulness)
        n_hallucinations = len(scenario_data[scenario_data['Faithfulness'] <= 2.0])
        rate = (n_hallucinations / n_total) * 100 if n_total > 0 else 0
        std_dev = scenario_data['Faithfulness'].std()
        
        risk_stats.append({
            "Scenario": scenario,
            "Hallucinations": n_hallucinations,
            "Total": n_total,
            "Hallucination Rate (%)": f"{rate:.1f}%",
            "Std Dev (Consistency)": f"{std_dev:.3f}"
        })

    risk_df = pd.DataFrame(risk_stats)
    
    return {
        "abstain_stats": abstain_stats,
        "score_stats": score_stats,
        "risk_stats": risk_df
    }
