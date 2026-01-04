"""
Medical QA - RAG Pipeline Module
=============================================

This module implements the Retrieval-Augmented Generation (RAG) pipeline.
It encapsulates the logic for:
1. Loading the Knowledge Base (Parquet) and FAISS Index.
2. Loading the Embedding Model (SentenceTransformers) and LLM (Gemma 2).
3. Retrieving relevant documents based on semantic similarity.
4. Generating answers with "Abstain" logic for low-confidence retrieval.

Classes:
    RAGPipeline: Main class to handle the RAG process.
"""

from typing import List, Dict, Tuple, Optional, Union
import torch
import faiss
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# Import project configuration
from src.config import (
    MODEL_NAME, 
    EMBEDDING_MODEL_NAME, 
    INDEX_PATH, 
    PROCESSED_DATA_PATH, 
    TOP_K
)

class RAGPipeline:
    """
    A class to manage the Medical RAG Pipeline.
    
    Attributes:
        device (str): The device to run models on ('cuda' or 'cpu').
        embed_model (SentenceTransformer): Model for generating text embeddings.
        index (faiss.Index): FAISS index for fast similarity search.
        df (pd.DataFrame): The knowledge base containing text chunks.
        model (AutoModelForCausalLM): The quantized LLM for generation.
        tokenizer (AutoTokenizer): Tokenizer for the LLM.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the pipeline configuration.
        
        Args:
            device (str, optional): 'cuda' or 'cpu'. Defaults to auto-detection.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Resources (loaded lazily via load_resources)
        self.embed_model = None
        self.index = None
        self.df = None
        self.model = None
        self.tokenizer = None
        
    def load_resources(self) -> None:
        """
        Loads all necessary heavy resources into memory.
        
        This includes:
        - Knowledge Base (Parquet file)
        - FAISS Index (Vector DB)
        - Embedding Model (SentenceTransformer)
        - Large Language Model (4-bit Quantized Gemma 2)
        """
        print(f"--- Loading RAG Resources on {self.device.upper()} ---")

        # 1. Load Knowledge Base Data
        print(f"Loading Knowledge Base from: {PROCESSED_DATA_PATH}")
        self.df = pd.read_parquet(PROCESSED_DATA_PATH)
        
        # 2. Load FAISS Index
        print(f"Loading FAISS Index from: {INDEX_PATH}")
        self.index = faiss.read_index(INDEX_PATH)
        
        # 3. Load Embedding Model
        print(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
        
        # 4. Load LLM (Quantized)
        print(f"Loading LLM: {MODEL_NAME}")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
            # Removed CPU offload to force GPU usage for performance
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map=self.device # Force specific device (e.g., 'cuda')
        )
        
        print("All resources loaded successfully.\n")

    def retrieve(self, query: str, k: int = TOP_K) -> List[Dict]:
        """
        Retrieves the top-k most relevant documents for a given query.
        
        Args:
            query (str): The user's question.
            k (int): Number of documents to retrieve.
            
        Returns:
            list[dict]: A list of dictionaries containing 'text', 'id', and 'distance'.
            
        Raises:
            ValueError: If resources are not loaded.
        """
        if self.embed_model is None:
            raise ValueError("Pipeline not loaded. Call load_resources() first.")
            
        # 1. Encode the query
        query_embedding = self.embed_model.encode([query], convert_to_numpy=True)
        
        # 2. Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # 3. Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.df): # Safety check for index bounds
                results.append({
                    'text': self.df.iloc[idx]['text'],
                    'id': self.df.iloc[idx]['chunk_id'],
                    'distance': distances[0][i] # L2 Distance (Lower is better)
                })
                
        return results

    def generate_answer(self, query: str, k: int = TOP_K, distance_threshold: float = 0.80, safe_prompt: bool = True) -> Tuple[str, List[Dict]]:
        """
        Generates an answer using the RAG approach with reliability checks.
        
        Logic:
        1. Retrieve relevant documents.
        2. Check if the best document is close enough (distance < threshold).
           - If NOT, return "I don't know" (Abstain).
        3. Construct a prompt with the retrieved context.
        4. Generate the answer using the LLM.
        
        Args:
            query (str): The user's question.
            k (int): Number of context documents to use.
            distance_threshold (float): Max L2 distance to consider a match valid.
            safe_prompt (bool): If True, instructs the model to say "I don't know" if context is missing.
            
        Returns:
            tuple: (answer_string, list_of_source_documents)
        """
        # --- Step 1: Retrieval ---
        retrieved_docs = self.retrieve(query, k)
        
        # --- Step 2: Reliability Check (Abstain Logic) ---
        if not retrieved_docs:
            return "I don't know (No documents found).", []
            
        best_distance = retrieved_docs[0]['distance']
        
        # If the best match is too far away, we assume the model doesn't know.
        if best_distance > distance_threshold:
            return (
                f"I don't know. (Confidence too low: {best_distance:.2f} > {distance_threshold})", 
                retrieved_docs
            )

        # --- Step 3: Prompt Construction ---
        # Combine retrieved texts into a single context block
        context_text = "\n\n".join([f"Doc {i+1}: {d['text']}" for i, d in enumerate(retrieved_docs)])
        
        if safe_prompt:
            instruction = "Answer the question based ONLY on the provided context."
            safety_instruction = 'If the answer is not in the context, say "I don\'t know".'
        else:
            # Naive mode: Less restrictive, prone to using internal knowledge (Leaky RAG)
            instruction = "Answer the question using the provided context."
            safety_instruction = ""

        messages = [
            {
                "role": "user", 
                "content": f"""You are a helpful medical assistant. {instruction}
Answer in a complete sentence.
{safety_instruction}

Context:
{context_text}

Question: {query}"""
            }
        ]

        # --- Step 4: Generation ---
        # Apply chat template (handles special tokens like <start_of_turn>)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=512, # Increased to prevent truncation
                do_sample=False, # Deterministic generation
                temperature=0.0
            )
        
        # Decode only the new tokens (the answer)
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        answer_only = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return answer_only.strip(), retrieved_docs
