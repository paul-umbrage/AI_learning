"""
PDF processing utilities for extracting text, chunking, and preparing for embedding
"""
import pdfplumber
import fitz  # PyMuPDF
import os
from typing import List, Dict, Optional, Any
import tiktoken
from contextual_chunking import contextual_chunk, ContentType

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, any]]:
    """
    Extract text from PDF with page numbers
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        List of dictionaries with 'page_number' and 'text' keys
    """
    pages_data = []
    
    try:
        # Use pdfplumber for better text extraction
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    pages_data.append({
                        'page_number': page_num,
                        'text': text.strip()
                    })
    except Exception as e:
        print(f"Error with pdfplumber, trying PyMuPDF: {e}")
        # Fallback to PyMuPDF
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text and text.strip():
                pages_data.append({
                    'page_number': page_num + 1,
                    'text': text.strip()
                })
        doc.close()
    
    return pages_data

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks with overlap
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk (in characters)
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary if possible
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                last_punct = text.rfind(punct, start, end)
                if last_punct != -1:
                    end = last_punct + len(punct)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def chunk_text_by_tokens(text: str, max_tokens: int = 500, overlap_tokens: int = 50) -> List[str]:
    """
    Split text into chunks based on token count (more accurate for embeddings)
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap
    
    Returns:
        List of text chunks
    """
    encoding = tiktoken.get_encoding("cl100k_base")  # Used by GPT-3.5 and GPT-4
    
    # Encode text to tokens
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + max_tokens
        
        # Decode chunk
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        
        # Move start with overlap
        start = end - overlap_tokens
        if start >= len(tokens):
            break
    
    return chunks

def process_pdf_to_chunks(
    pdf_path: str, 
    chunk_method: str = "tokens", 
    chunk_size: int = 500, 
    overlap: int = 50,
    use_contextual_chunking: bool = False,
    contextual_method: str = "semantic",
    similarity_threshold: float = 0.7,
    client: Optional[Any] = None
) -> List[Dict[str, any]]:
    """
    Process PDF file and return chunks with metadata
    
    Args:
        pdf_path: Path to the PDF file
        chunk_method: "tokens", "characters", "semantic", or "sentence"
        chunk_size: Size of chunks (tokens or characters depending on method)
        overlap: Overlap size
        use_contextual_chunking: Enable contextual chunking (content-type aware)
        contextual_method: "semantic" or "sentence" (only if use_contextual_chunking=True)
        similarity_threshold: Threshold for semantic breaks (0.0-1.0)
        client: Optional OpenAI client for semantic chunking
    
    Returns:
        List of dictionaries with chunk data including:
            - filename: str
            - page_number: int
            - chunk_index: int
            - chunk_text: str
            - content_type: str (if contextual chunking enabled)
    """
    filename = os.path.basename(pdf_path)
    pages_data = extract_text_from_pdf(pdf_path)
    
    all_chunks = []
    
    for page_data in pages_data:
        page_number = page_data['page_number']
        text = page_data['text']
        
        # Use contextual chunking if enabled
        if use_contextual_chunking:
            contextual_chunks = contextual_chunk(
                text=text,
                chunk_method=contextual_method,
                max_tokens=chunk_size,
                base_overlap=overlap,
                similarity_threshold=similarity_threshold,
                client=client
            )
            
            # Add metadata to each chunk
            for chunk_index, chunk_data in enumerate(contextual_chunks):
                all_chunks.append({
                    'filename': filename,
                    'page_number': page_number,
                    'chunk_index': chunk_index,
                    'chunk_text': chunk_data['chunk_text'],
                    'content_type': chunk_data.get('content_type', 'paragraph')
                })
        else:
            # Use traditional chunking
            if chunk_method == "tokens":
                chunks = chunk_text_by_tokens(text, max_tokens=chunk_size, overlap_tokens=overlap)
            elif chunk_method == "sentence":
                # Use sentence-aware chunking
                from contextual_chunking import chunk_by_sentences
                chunks = chunk_by_sentences(text, max_tokens=chunk_size, overlap_sentences=max(1, overlap // 20))
            else:
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            
            # Add metadata to each chunk
            for chunk_index, chunk_text in enumerate(chunks):
                all_chunks.append({
                    'filename': filename,
                    'page_number': page_number,
                    'chunk_index': chunk_index,
                    'chunk_text': chunk_text
                })
    
    return all_chunks

