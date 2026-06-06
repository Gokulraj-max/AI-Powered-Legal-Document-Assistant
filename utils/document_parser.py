import os
from pypdf import PdfReader
from docx import Document

def split_text_into_chunks(text, chunk_size, chunk_overlap):
    """
    Splits text into chunks using a character-based sliding window.
    Attempts to split at sentence boundaries (. ) or word boundaries (space) to maintain readability.
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        
        if end < text_len:
            # Look back in a window of 150 characters for a clean split boundary
            window_start = max(start + 50, end - 150)
            search_window = text[window_start:end]
            
            # Try to split at a sentence end followed by whitespace
            last_period = -1
            for separator in [".\n", ". ", "? ", "! "]:
                pos = search_window.rfind(separator)
                if pos > last_period:
                    last_period = pos + 1 # Include the period
            
            if last_period != -1:
                end = window_start + last_period
            else:
                # Try to split at a newline
                last_newline = search_window.rfind("\n")
                if last_newline != -1:
                    end = window_start + last_newline
                else:
                    # Try to split at a space
                    last_space = search_window.rfind(" ")
                    if last_space != -1:
                        end = window_start + last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        # Move start pointer forward
        start = end - chunk_overlap
        if start >= text_len - 50: # If remaining is very small, stop
            break
            
    return chunks

def parse_pdf(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Extracts text page-by-page from a PDF and chunks each page.
    This guarantees page citations are 100% accurate.
    """
    chunks = []
    reader = PdfReader(file_path)
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        text = page.extract_text()
        if not text:
            continue
        
        # Clean text
        text = text.strip()
        if not text:
            continue
            
        page_chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "page": page_num,
                "location": f"Page {page_num}"
            })
    return chunks

def parse_docx(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Extracts text from DOCX paragraphs and chunks them.
    """
    doc = Document(file_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)
    
    raw_chunks = split_text_into_chunks(full_text, chunk_size, chunk_overlap)
    chunks = []
    for idx, chunk in enumerate(raw_chunks):
        chunks.append({
            "text": chunk,
            "page": None,
            "location": f"Section {idx + 1}"
        })
    return chunks

def parse_txt(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Extracts text from a TXT file with encoding fallbacks and chunks it.
    """
    content = ""
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
            
    if not content:
        raise ValueError("Unable to read TXT file. Encoding not supported.")
        
    raw_chunks = split_text_into_chunks(content.strip(), chunk_size, chunk_overlap)
    chunks = []
    for idx, chunk in enumerate(raw_chunks):
        chunks.append({
            "text": chunk,
            "page": None,
            "location": f"Paragraph {idx + 1}"
        })
    return chunks

def parse_document(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Routes parsing based on file extension.
    """
    _, ext = os.path.splitext(file_path.lower())
    if ext == ".pdf":
        return parse_pdf(file_path, chunk_size, chunk_overlap)
    elif ext == ".docx":
        return parse_docx(file_path, chunk_size, chunk_overlap)
    elif ext in [".txt", ".md"]:
        return parse_txt(file_path, chunk_size, chunk_overlap)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
