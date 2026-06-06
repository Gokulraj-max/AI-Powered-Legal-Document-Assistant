import sqlite3
import numpy as np
import os

class VectorStore:
    def __init__(self, db_path="C:\\Projects\\AI-Powered-Legal-Document-Assistant\\legal_assistant.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_connection() as conn:
            # Create documents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL UNIQUE,
                    file_type TEXT NOT NULL,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Create chunks table with cascade delete
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    page_num INTEGER,
                    location TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def add_document(self, filename, file_type, chunks_data):
        """
        Adds a document and its chunks with embeddings to the vector store.
        chunks_data is a list of dictionaries, e.g.:
        [
            {
                "text": "chunk text...",
                "page": 1,
                "location": "Page 1",
                "embedding": [0.1, 0.2, ...] (list or numpy array)
            },
            ...
        ]
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Insert document, if already exists overwrite or raise
                cursor.execute(
                    "INSERT INTO documents (filename, file_type) VALUES (?, ?)",
                    (filename, file_type)
                )
                doc_id = cursor.lastrowid
                
                # Insert chunks
                for idx, chunk in enumerate(chunks_data):
                    emb = np.array(chunk["embedding"], dtype=np.float32)
                    emb_blob = emb.tobytes()
                    cursor.execute(
                        """
                        INSERT INTO chunks (doc_id, chunk_index, text, page_num, location, embedding)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (doc_id, idx, chunk["text"], chunk["page"], chunk["location"], emb_blob)
                    )
                conn.commit()
                return doc_id
            except sqlite3.IntegrityError:
                # Document already exists, rollback and raise
                conn.rollback()
                raise ValueError(f"Document '{filename}' already exists in the database.")

    def delete_document(self, filename):
        """
        Deletes a document and cascades deletes all its chunks.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE filename = ?", (filename,))
            conn.commit()
            return cursor.rowcount > 0

    def get_all_documents(self):
        """
        Returns all registered documents in the system.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename, file_type, upload_date FROM documents ORDER BY upload_date DESC")
            rows = cursor.fetchall()
            return [{"filename": r[0], "file_type": r[1], "upload_date": r[2]} for r in rows]

    def get_stats(self):
        """
        Returns basic stats about the database.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            return {
                "total_documents": doc_count,
                "total_chunks": chunk_count
            }

    def search(self, query_embedding, top_k=5, min_score=0.0):
        """
        Performs in-memory cosine similarity search.
        """
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        # Fetch all chunks
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT chunks.text, chunks.page_num, chunks.location, chunks.embedding, documents.filename
                FROM chunks
                JOIN documents ON chunks.doc_id = documents.id
            """)
            rows = cursor.fetchall()

        if not rows:
            return []

        results = []
        for text, page_num, location, emb_blob, filename in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                score = 0.0
            else:
                score = float(np.dot(emb, q) / (emb_norm * q_norm))
            
            if score >= min_score:
                results.append({
                    "text": text,
                    "page": page_num,
                    "location": location,
                    "filename": filename,
                    "score": score
                })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def reset_database(self):
        """
        Wipes out all documents and chunks.
        """
        with self._get_connection() as conn:
            conn.execute("DELETE FROM documents")
            conn.commit()
