"""
Hybrid search implementation combining vector similarity with keyword/BM25 search.

This module provides:
- Keyword-based search (using PostgreSQL full-text search or simple matching)
- Score combination (weighted combination of vector + keyword scores)
- Metadata filtering support
"""

import re
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter
import math


def extract_keywords(query: str, min_length: int = 2) -> List[str]:
    """
    Extract keywords from query string.
    
    Args:
        query: Query string
        min_length: Minimum keyword length
    
    Returns:
        List of keywords (lowercased, no stopwords)
    """
    # Simple stopwords list (can be expanded)
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
        'that', 'these', 'those', 'what', 'which', 'who', 'when', 'where',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'now'
    }
    
    # Extract words (alphanumeric only)
    words = re.findall(r'\b\w+\b', query.lower())
    
    # Filter out stopwords and short words
    keywords = [w for w in words if len(w) >= min_length and w not in stopwords]
    
    return keywords


def calculate_keyword_score(chunk_text: str, keywords: List[str]) -> float:
    """
    Calculate keyword match score for a chunk.
    
    Uses a simple TF-IDF-like scoring:
    - Term frequency (TF): how often keywords appear in chunk
    - Inverse document frequency (IDF): penalize common words
    
    Args:
        chunk_text: Text chunk to score
        keywords: List of keywords to match
    
    Returns:
        Score between 0.0 and 1.0
    """
    if not keywords:
        return 0.0
    
    chunk_lower = chunk_text.lower()
    chunk_words = re.findall(r'\b\w+\b', chunk_lower)
    
    if not chunk_words:
        return 0.0
    
    # Count keyword occurrences
    keyword_counts = Counter()
    for keyword in keywords:
        # Count exact matches
        count = chunk_lower.count(keyword)
        keyword_counts[keyword] = count
    
    # Calculate TF (term frequency) - normalized by chunk length
    total_matches = sum(keyword_counts.values())
    if total_matches == 0:
        return 0.0
    
    # TF: term frequency normalized by document length
    tf = total_matches / len(chunk_words)
    
    # Simple IDF approximation: penalize very common keywords
    # In a full implementation, you'd calculate IDF from corpus statistics
    # For now, we use a simple heuristic: longer keywords are more specific
    idf_weights = {}
    for keyword in keywords:
        # Longer keywords are more specific (higher IDF)
        idf_weights[keyword] = 1.0 + (len(keyword) - 3) * 0.1 if len(keyword) > 3 else 1.0
    
    # Calculate weighted score
    weighted_score = sum(
        (keyword_counts[k] / len(chunk_words)) * idf_weights.get(k, 1.0)
        for k in keywords
    )
    
    # Normalize to 0-1 range (using sigmoid-like function)
    score = min(1.0, weighted_score * 2.0)
    
    return score


def search_keywords(
    query: str,
    chunks: List[Tuple[str, str, int, float]],
    top_k: Optional[int] = None
) -> List[Tuple[str, str, int, float]]:
    """
    Search chunks by keyword matching.
    
    Args:
        query: Query string
        chunks: List of (chunk_text, filename, page_number, vector_score) tuples
        top_k: Number of results to return (None = all)
    
    Returns:
        List of (chunk_text, filename, page_number, keyword_score) tuples
    """
    keywords = extract_keywords(query)
    
    if not keywords:
        # If no keywords, return chunks with zero keyword scores
        return [(chunk[0], chunk[1], chunk[2], 0.0) for chunk in chunks]
    
    # Calculate keyword scores
    scored_chunks = []
    for chunk_text, filename, page_number, _ in chunks:
        keyword_score = calculate_keyword_score(chunk_text, keywords)
        scored_chunks.append((chunk_text, filename, page_number, keyword_score))
    
    # Sort by keyword score (descending)
    scored_chunks.sort(key=lambda x: x[3], reverse=True)
    
    # Return top_k if specified
    if top_k:
        return scored_chunks[:top_k]
    
    return scored_chunks


def combine_scores(
    vector_score: float,
    keyword_score: float,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> float:
    """
    Combine vector similarity score with keyword score.
    
    Args:
        vector_score: Vector similarity score (0.0-1.0)
        keyword_score: Keyword match score (0.0-1.0)
        vector_weight: Weight for vector score (default: 0.7)
        keyword_weight: Weight for keyword score (default: 0.3)
    
    Returns:
        Combined score (0.0-1.0)
    """
    # Normalize weights
    total_weight = vector_weight + keyword_weight
    if total_weight == 0:
        return 0.0
    
    normalized_vector_weight = vector_weight / total_weight
    normalized_keyword_weight = keyword_weight / total_weight
    
    # Weighted combination
    combined_score = (
        normalized_vector_weight * vector_score +
        normalized_keyword_weight * keyword_score
    )
    
    return min(1.0, max(0.0, combined_score))


def hybrid_search(
    query: str,
    vector_results: List[Tuple[str, str, int, float]],
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    top_k: Optional[int] = None
) -> List[Tuple[str, str, int, float]]:
    """
    Perform hybrid search combining vector and keyword scores.
    
    Args:
        query: Query string
        vector_results: List of (chunk_text, filename, page_number, vector_score) tuples
        vector_weight: Weight for vector similarity (default: 0.7)
        keyword_weight: Weight for keyword matching (default: 0.3)
        top_k: Number of results to return (None = all)
    
    Returns:
        List of (chunk_text, filename, page_number, combined_score) tuples
    """
    if not vector_results:
        return []
    
    # Calculate keyword scores for all chunks
    keyword_results = search_keywords(query, vector_results)
    
    # Create a dictionary for quick lookup
    keyword_scores = {
        (chunk[0], chunk[1], chunk[2]): chunk[3]
        for chunk in keyword_results
    }
    
    # Combine scores
    combined_results = []
    for chunk_text, filename, page_number, vector_score in vector_results:
        keyword_score = keyword_scores.get((chunk_text, filename, page_number), 0.0)
        combined_score = combine_scores(
            vector_score,
            keyword_score,
            vector_weight,
            keyword_weight
        )
        combined_results.append((chunk_text, filename, page_number, combined_score))
    
    # Sort by combined score (descending)
    combined_results.sort(key=lambda x: x[3], reverse=True)
    
    # Return top_k if specified
    if top_k:
        return combined_results[:top_k]
    
    return combined_results


def search_with_metadata_filter(
    query: str,
    vector_results: List[Tuple[str, str, int, float]],
    filename: Optional[str] = None,
    page_range: Optional[Tuple[int, int]] = None,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    top_k: Optional[int] = None
) -> List[Tuple[str, str, int, float]]:
    """
    Perform hybrid search with metadata filtering.
    
    Args:
        query: Query string
        vector_results: List of (chunk_text, filename, page_number, vector_score) tuples
        filename: Optional filename filter
        page_range: Optional (min_page, max_page) tuple
        vector_weight: Weight for vector similarity
        keyword_weight: Weight for keyword matching
        top_k: Number of results to return
    
    Returns:
        Filtered and scored results
    """
    # Apply metadata filters
    filtered_results = []
    for chunk_text, chunk_filename, page_number, vector_score in vector_results:
        # Filename filter
        if filename and chunk_filename != filename:
            continue
        
        # Page range filter
        if page_range:
            min_page, max_page = page_range
            if not (min_page <= page_number <= max_page):
                continue
        
        filtered_results.append((chunk_text, chunk_filename, page_number, vector_score))
    
    # Perform hybrid search on filtered results
    return hybrid_search(
        query,
        filtered_results,
        vector_weight,
        keyword_weight,
        top_k
    )

