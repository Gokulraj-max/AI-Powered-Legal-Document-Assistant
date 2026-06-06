from google import genai
from google.genai import types
import numpy as np

def get_client(api_key):
    """
    Initializes and returns the GenAI client.
    """
    if not api_key or not api_key.strip():
        raise ValueError("Gemini API Key is missing. Please provide it in the sidebar.")
    return genai.Client(api_key=api_key.strip())

def get_embedding(api_key, text):
    """
    Generates embedding for a single text using text-embedding-004.
    """
    client = get_client(api_key)
    try:
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        return response.embeddings[0].values
    except Exception as e:
        raise RuntimeError(f"Error generating embedding: {e}")

def get_embeddings_batch(api_key, texts, batch_size=100):
    """
    Generates embeddings for a list of texts in batches.
    """
    client = get_client(api_key)
    embeddings = []
    
    # Process in batches to avoid API limits
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=batch
            )
            for emb in response.embeddings:
                embeddings.append(emb.values)
        except Exception as e:
            raise RuntimeError(f"Error generating embedding batch starting at index {i}: {e}")
            
    return embeddings

def generate_answer(api_key, query, retrieved_chunks, model_name="gemini-2.5-flash", temperature=0.2):
    """
    Assembles prompt with retrieved chunks and queries the Gemini LLM for an answer.
    """
    client = get_client(api_key)
    
    # Format the context text with clear citation markers
    context_str = ""
    for idx, chunk in enumerate(retrieved_chunks):
        context_str += f"--- CONTEXT CHUNK {idx + 1} ---\n"
        context_str += f"Source: {chunk['filename']}\n"
        context_str += f"Location: {chunk['location']}\n"
        context_str += f"Content: {chunk['text']}\n\n"
        
    # Build a strict legal-focused prompt
    system_instruction = (
        "You are an expert legal assistant. Your task is to answer the user's question "
        "accurately and professionally based ONLY on the provided context chunks. "
        "Strictly adhere to these rules:\n"
        "1. Base your answer solely on the provided context chunks. Do not assume, hallucinate, "
        "or bring in external knowledge.\n"
        "2. For every fact, claim, or quote you provide, cite the source file name and page/location "
        "(e.g., '[Document_A.pdf, Page 3]' or '[Contract_Draft.docx, Section 2]') at the end of the sentence or paragraph.\n"
        "3. If the context does not contain enough information to answer the question, state clearly: "
        "'Based on the uploaded documents, I cannot find the answer to this question.' Do not attempt to guess.\n"
        "4. Keep your tone objective, precise, and professional, suitable for a legal context."
    )
    
    prompt = (
        f"Query: {query}\n\n"
        f"Retrieved Document Context:\n{context_str}\n"
        f"Please answer the query following the system instructions."
    )
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=2048
            )
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Error generating answer: {e}")
