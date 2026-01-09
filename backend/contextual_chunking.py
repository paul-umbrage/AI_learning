"""
Contextual chunking for improved text segmentation.

Provides semantic chunking, sentence-aware chunking, content-type
optimization, and special handling for structured content.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import tiktoken
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class ContentType(Enum):
    """Content type classification"""
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    HEADING = "heading"
    CODE = "code"
    MIXED = "mixed"


def detect_content_type(text: str) -> ContentType:
    """
    Detect the type of content in a text segment.
    
    Args:
        text: Text segment to analyze
    
    Returns:
        ContentType enum
    """
    text_stripped = text.strip()
    
    # Check for lists (bulleted or numbered)
    list_patterns = [
        r'^\s*[-•*]\s+',  # Bullet points
        r'^\s*\d+[.)]\s+',  # Numbered lists
        r'^\s*[a-z][.)]\s+',  # Lettered lists
    ]
    list_lines = sum(1 for line in text_stripped.split('\n') 
                     if any(re.match(pattern, line) for pattern in list_patterns))
    if list_lines >= 2:
        return ContentType.LIST
    
    # Check for tables (multiple columns separated by spaces/tabs)
    lines = text_stripped.split('\n')
    if len(lines) >= 3:
        # Check if lines have consistent column structure
        column_counts = [len(re.split(r'\s{2,}|\t', line.strip())) for line in lines[:5]]
        if len(set(column_counts)) == 1 and column_counts[0] >= 2:
            return ContentType.TABLE
    
    # Check for headings (short lines, all caps, or with special formatting)
    if len(text_stripped.split('\n')) == 1 and len(text_stripped) < 100:
        if text_stripped.isupper() or text_stripped.startswith('#'):
            return ContentType.HEADING
    
    # Check for code (high density of special characters)
    code_chars = sum(1 for c in text_stripped if c in '{}[]();=<>')
    if code_chars > len(text_stripped) * 0.1:
        return ContentType.CODE
    
    # Default to paragraph
    return ContentType.PARAGRAPH


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using improved sentence detection.
    
    Args:
        text: Text to split
    
    Returns:
        List of sentences
    """
    # Pattern to match sentence endings
    # Handles: . ! ? followed by space/newline/capital letter
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+(?=[A-Z])'
    
    sentences = re.split(sentence_pattern, text)
    
    # Clean up sentences
    cleaned = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:  # Filter very short fragments
            cleaned.append(sentence)
    
    return cleaned if cleaned else [text]


def chunk_by_sentences(
    text: str,
    max_tokens: int = 500,
    overlap_sentences: int = 2,
    encoding: Optional[Any] = None
) -> List[str]:
    """
    Chunk text by sentences, respecting token limits.
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_sentences: Number of sentences to overlap between chunks
        encoding: Optional tiktoken encoding (creates if not provided)
    
    Returns:
        List of sentence-aware chunks
    """
    if not encoding:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    sentences = split_into_sentences(text)
    
    if not sentences:
        return [text]
    
    chunks = []
    current_chunk_sentences = []
    current_tokens = 0
    
    for i, sentence in enumerate(sentences):
        sentence_tokens = len(encoding.encode(sentence))
        
        # If adding this sentence would exceed limit, finalize current chunk
        if current_tokens + sentence_tokens > max_tokens and current_chunk_sentences:
            # Create chunk from current sentences
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap
            overlap_start = max(0, len(current_chunk_sentences) - overlap_sentences)
            current_chunk_sentences = current_chunk_sentences[overlap_start:]
            current_tokens = sum(len(encoding.encode(s)) for s in current_chunk_sentences)
        
        # Add sentence to current chunk
        current_chunk_sentences.append(sentence)
        current_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk_sentences:
        chunk_text = ' '.join(current_chunk_sentences)
        chunks.append(chunk_text)
    
    return chunks if chunks else [text]


def calculate_semantic_similarity_batch(
    texts: List[str],
    client: Optional[OpenAI] = None
) -> np.ndarray:
    """
    Calculate pairwise semantic similarities between text segments.
    
    Args:
        texts: List of text segments
        client: Optional OpenAI client
    
    Returns:
        NxN similarity matrix
    """
    if not client:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Fallback: return identity matrix (no semantic info)
            return np.eye(len(texts))
        client = OpenAI(api_key=api_key)
    
    try:
        # Get embeddings for all texts
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        
        embeddings = [item.embedding for item in response.data]
        embeddings_array = np.array(embeddings)
        
        # Calculate cosine similarity matrix
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        normalized = embeddings_array / (norms + 1e-8)  # Avoid division by zero
        similarity_matrix = np.dot(normalized, normalized.T)
        
        return similarity_matrix
    except Exception:
        # Fallback: return identity matrix
        return np.eye(len(texts))


def find_semantic_breaks(
    sentences: List[str],
    similarity_threshold: float = 0.7,
    client: Optional[OpenAI] = None
) -> List[int]:
    """
    Find natural break points using semantic similarity.
    
    Lower similarity between adjacent sentences indicates a topic change.
    
    Args:
        sentences: List of sentences
        similarity_threshold: Threshold below which to consider a break
        client: Optional OpenAI client
    
    Returns:
        List of indices where breaks should occur (before this sentence)
    """
    if len(sentences) <= 1:
        return []
    
    # Calculate similarities between adjacent sentences
    breaks = []
    
    # Process in batches to avoid too many API calls
    batch_size = 50
    for i in range(0, len(sentences) - 1, batch_size):
        batch_sentences = sentences[i:min(i + batch_size + 1, len(sentences))]
        
        try:
            similarity_matrix = calculate_semantic_similarity_batch(batch_sentences, client)
            
            # Check similarities between adjacent sentences
            for j in range(len(batch_sentences) - 1):
                similarity = similarity_matrix[j, j + 1]
                if similarity < similarity_threshold:
                    breaks.append(i + j + 1)  # Break before sentence j+1
        except Exception:
            # If semantic analysis fails, don't add breaks
            continue
    
    return sorted(set(breaks))


def semantic_chunk(
    text: str,
    max_tokens: int = 500,
    similarity_threshold: float = 0.7,
    min_chunk_tokens: int = 100,
    client: Optional[OpenAI] = None,
    encoding: Optional[Any] = None
) -> List[str]:
    """
    Chunk text using semantic similarity to find natural breaks.
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        similarity_threshold: Similarity threshold for detecting breaks
        min_chunk_tokens: Minimum tokens per chunk
        client: Optional OpenAI client
        encoding: Optional tiktoken encoding
    
    Returns:
        List of semantically-aware chunks
    """
    if not encoding:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    sentences = split_into_sentences(text)
    
    if not sentences:
        return [text]
    
    # Find semantic breaks
    break_indices = find_semantic_breaks(sentences, similarity_threshold, client)
    
    # Create chunks respecting semantic breaks and token limits
    chunks = []
    current_chunk_sentences = []
    current_tokens = 0
    sentence_idx = 0
    
    while sentence_idx < len(sentences):
        sentence = sentences[sentence_idx]
        sentence_tokens = len(encoding.encode(sentence))
        
        # Check if we should break here (semantic break or token limit)
        should_break = (
            sentence_idx in break_indices or
            (current_tokens + sentence_tokens > max_tokens and current_chunk_sentences)
        )
        
        if should_break and current_chunk_sentences:
            # Finalize current chunk if it meets minimum size
            chunk_text = ' '.join(current_chunk_sentences)
            chunk_tokens = len(encoding.encode(chunk_text))
            
            if chunk_tokens >= min_chunk_tokens:
                chunks.append(chunk_text)
                current_chunk_sentences = []
                current_tokens = 0
            # If chunk is too small, continue adding sentences
        
        # Add sentence to current chunk
        current_chunk_sentences.append(sentence)
        current_tokens += sentence_tokens
        
        # Force break if chunk is too large
        if current_tokens > max_tokens * 1.2:  # Allow 20% overflow
            chunk_text = ' '.join(current_chunk_sentences[:-1])  # Exclude last sentence
            if chunk_text.strip():
                chunks.append(chunk_text)
            current_chunk_sentences = [sentence]  # Start new chunk with current sentence
            current_tokens = sentence_tokens
        
        sentence_idx += 1
    
    # Add final chunk
    if current_chunk_sentences:
        chunk_text = ' '.join(current_chunk_sentences)
        chunks.append(chunk_text)
    
    return chunks if chunks else [text]


def get_optimal_overlap(content_type: ContentType, base_overlap: int = 50) -> int:
    """
    Get optimal overlap based on content type.
    
    Args:
        content_type: Type of content
        base_overlap: Base overlap in tokens
    
    Returns:
        Optimal overlap in tokens
    """
    overlap_multipliers = {
        ContentType.PARAGRAPH: 1.0,  # Standard overlap
        ContentType.LIST: 0.5,  # Less overlap for lists (items are more independent)
        ContentType.TABLE: 0.3,  # Minimal overlap for tables (preserve structure)
        ContentType.HEADING: 2.0,  # More overlap around headings (important context)
        ContentType.CODE: 0.2,  # Minimal overlap for code (preserve syntax)
        ContentType.MIXED: 1.0
    }
    
    multiplier = overlap_multipliers.get(content_type, 1.0)
    return int(base_overlap * multiplier)


def chunk_list_content(text: str, max_tokens: int = 500, overlap_items: int = 1) -> List[str]:
    """
    Chunk list content preserving list items.
    
    Args:
        text: List content to chunk
        max_tokens: Maximum tokens per chunk
        overlap_items: Number of list items to overlap
    
    Returns:
        List of chunks preserving list structure
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Split by list item markers
    list_item_pattern = r'(?=^\s*[-•*]\s+|^\s*\d+[.)]\s+|^\s*[a-z][.)]\s+)'
    items = re.split(list_item_pattern, text, flags=re.MULTILINE)
    items = [item.strip() for item in items if item.strip()]
    
    if not items:
        return [text]
    
    chunks = []
    current_chunk_items = []
    current_tokens = 0
    
    for i, item in enumerate(items):
        item_tokens = len(encoding.encode(item))
        
        if current_tokens + item_tokens > max_tokens and current_chunk_items:
            # Finalize chunk
            chunk_text = '\n'.join(current_chunk_items)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap
            overlap_start = max(0, len(current_chunk_items) - overlap_items)
            current_chunk_items = current_chunk_items[overlap_start:]
            current_tokens = sum(len(encoding.encode(item)) for item in current_chunk_items)
        
        current_chunk_items.append(item)
        current_tokens += item_tokens
    
    if current_chunk_items:
        chunk_text = '\n'.join(current_chunk_items)
        chunks.append(chunk_text)
    
    return chunks if chunks else [text]


def chunk_table_content(text: str, max_tokens: int = 500) -> List[str]:
    """
    Chunk table content preserving table structure.
    
    Args:
        text: Table content to chunk
        max_tokens: Maximum tokens per chunk
    
    Returns:
        List of chunks preserving table rows
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    
    lines = text.split('\n')
    if not lines:
        return [text]
    
    chunks = []
    current_chunk_lines = []
    current_tokens = 0
    
    for line in lines:
        line_tokens = len(encoding.encode(line))
        
        if current_tokens + line_tokens > max_tokens and current_chunk_lines:
            # Finalize chunk (preserve header if possible)
            chunk_text = '\n'.join(current_chunk_lines)
            chunks.append(chunk_text)
            current_chunk_lines = []
            current_tokens = 0
        
        current_chunk_lines.append(line)
        current_tokens += line_tokens
    
    if current_chunk_lines:
        chunk_text = '\n'.join(current_chunk_lines)
        chunks.append(chunk_text)
    
    return chunks if chunks else [text]


def contextual_chunk(
    text: str,
    chunk_method: str = "semantic",
    max_tokens: int = 500,
    base_overlap: int = 50,
    similarity_threshold: float = 0.7,
    client: Optional[OpenAI] = None
) -> List[Dict[str, Any]]:
    """
    Contextually chunk text with content-type awareness.
    
    Args:
        text: Text to chunk
        chunk_method: "semantic", "sentence", or "tokens"
        max_tokens: Maximum tokens per chunk
        base_overlap: Base overlap in tokens
        similarity_threshold: Threshold for semantic breaks
        client: Optional OpenAI client
    
    Returns:
        List of chunk dictionaries with metadata:
            - chunk_text: str
            - content_type: ContentType
            - start_index: int (character position)
            - end_index: int
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Detect content type
    content_type = detect_content_type(text)
    
    # Choose chunking strategy based on content type
    if content_type == ContentType.LIST:
        chunk_texts = chunk_list_content(text, max_tokens, overlap_items=1)
    elif content_type == ContentType.TABLE:
        chunk_texts = chunk_table_content(text, max_tokens)
    elif chunk_method == "semantic":
        chunk_texts = semantic_chunk(
            text,
            max_tokens=max_tokens,
            similarity_threshold=similarity_threshold,
            client=client
        )
    elif chunk_method == "sentence":
        optimal_overlap = get_optimal_overlap(content_type, base_overlap)
        chunk_texts = chunk_by_sentences(
            text,
            max_tokens=max_tokens,
            overlap_sentences=max(1, optimal_overlap // 20),  # Approximate sentences
            encoding=encoding
        )
    else:
        # Fallback to token-based chunking
        tokens = encoding.encode(text)
        chunk_texts = []
        start = 0
        optimal_overlap = get_optimal_overlap(content_type, base_overlap)
        
        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            chunk_texts.append(chunk_text.strip())
            start = end - optimal_overlap
            if start >= len(tokens):
                break
    
    # Build result with metadata
    chunks = []
    current_pos = 0
    
    for chunk_text in chunk_texts:
        if not chunk_text.strip():
            continue
        
        # Find position in original text
        start_index = text.find(chunk_text[:50], current_pos)  # Find by first 50 chars
        if start_index == -1:
            start_index = current_pos
        end_index = start_index + len(chunk_text)
        current_pos = end_index
        
        chunks.append({
            'chunk_text': chunk_text,
            'content_type': content_type.value,
            'start_index': start_index,
            'end_index': end_index,
            'token_count': len(encoding.encode(chunk_text))
        })
    
    return chunks if chunks else [{
        'chunk_text': text,
        'content_type': content_type.value,
        'start_index': 0,
        'end_index': len(text),
        'token_count': len(encoding.encode(text))
    }]

