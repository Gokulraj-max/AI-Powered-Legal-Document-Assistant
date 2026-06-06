import os
import numpy as np
from utils.document_parser import split_text_into_chunks, parse_document
from utils.vector_store import VectorStore

def test_chunking():
    print("Testing text chunking sliding-window logic...")
    sample_text = (
        "This is sentence one. This is sentence two. This is sentence three. "
        "This is sentence four. This is sentence five. This is sentence six."
    )
    chunks = split_text_into_chunks(sample_text, chunk_size=50, chunk_overlap=15)
    print(f"Original length: {len(sample_text)}, generated {len(chunks)} chunks:")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i+1}: '{c}' (len={len(c)})")
    
    assert len(chunks) > 0, "Should generate at least one chunk"
    print("Chunking test PASSED.\n")

def test_vector_store():
    print("Testing SQLite + NumPy Vector Store...")
    db_path = "C:\\Projects\\AI-Powered-Legal-Document-Assistant\\test_legal.db"
    
    # Clean up test db if exists
    if os.path.exists(db_path):
        os.remove(db_path)
        
    store = VectorStore(db_path=db_path)
    
    # Insert mock document chunks
    mock_chunks = [
        {"text": "Contract govern the purchase of apples.", "page": 1, "location": "Page 1", "embedding": [0.9, 0.1, 0.0]},
        {"text": "Under no circumstances shall the supplier be liable for bananas.", "page": 2, "location": "Page 2", "embedding": [0.1, 0.9, 0.0]},
        {"text": "Governing law is of the State of New York.", "page": 3, "location": "Page 3", "embedding": [0.0, 0.0, 1.0]}
    ]
    
    store.add_document("mock_contract.txt", "txt", mock_chunks)
    
    # Test stats
    stats = store.get_stats()
    print(f"Stats after addition: {stats}")
    assert stats["total_documents"] == 1
    assert stats["total_chunks"] == 3
    
    # Test search for "apples"
    query_emb = [1.0, 0.0, 0.0]
    results = store.search(query_emb, top_k=2)
    print("Search results for query [1.0, 0.0, 0.0]:")
    for r in results:
        print(f"  Doc: {r['filename']} | Loc: {r['location']} | Score: {r['score']:.4f} | Text: '{r['text']}'")
        
    assert len(results) == 2
    assert "apples" in results[0]["text"]
    assert results[0]["score"] > 0.8
    
    # Test deletion
    store.delete_document("mock_contract.txt")
    stats = store.get_stats()
    print(f"Stats after deletion: {stats}")
    assert stats["total_documents"] == 0
    assert stats["total_chunks"] == 0
    
    # Clean up test db
    if os.path.exists(db_path):
        os.remove(db_path)
        
    print("Vector Store test PASSED.\n")

if __name__ == "__main__":
    print("--- RUNNING AUTOMATED VERIFICATION ---")
    test_chunking()
    test_vector_store()
    print("All local tests PASSED successfully!")
