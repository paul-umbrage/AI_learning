"""
Reranking utilities for improving retrieval quality.

This module provides different reranking strategies to filter and reorder
retrieved chunks based on relevance to the query.
"""

from typing import List, Tuple, Dict, Any
import re


def rerank_by_similarity_threshold(
    results: List[Tuple[str, str, int, float]],
    min_similarity: float = 0.7
) -> List[Tuple[str, str, int, float]]:
    """
    Filter results by minimum similarity threshold.
    
    Args:
        results: List of (chunk_text, filename, page_number, similarity) tuples
        min_similarity: Minimum similarity score to include (0.0-1.0)
    
    Returns:
        Filtered results above threshold
    """
    return [r for r in results if r[3] >= min_similarity]


def rerank_by_keyword_overlap(
    query: str,
    results: List[Tuple[str, str, int, float]],
    top_k: int = 3
) -> List[Tuple[str, str, int, float]]:
    """
    Rerank results by keyword overlap with query.
    
    This is a simple reranking strategy that boosts chunks containing
    query keywords. Useful for combining semantic similarity with exact matches.
    
    Args:
        query: Original query string
        results: List of (chunk_text, filename, page_number, similarity) tuples
        top_k: Number of results to return
    
    Returns:
        Reranked results
    """
    # Extract keywords from query (simple word-based)
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    
    scored_results = []
    for chunk_text, filename, page_number, similarity in results:
        # Count keyword matches
        chunk_words = set(re.findall(r'\b\w+\b', chunk_text.lower()))
        keyword_overlap = len(query_words & chunk_words)
        keyword_score = keyword_overlap / len(query_words) if query_words else 0
        
        # Combine similarity score with keyword score (weighted)
        combined_score = (similarity * 0.7) + (keyword_score * 0.3)
        
        scored_results.append((chunk_text, filename, page_number, combined_score))
    
    # Sort by combined score and return top_k
    scored_results.sort(key=lambda x: x[3], reverse=True)
    return scored_results[:top_k]


def rerank_by_diversity(
    results: List[Tuple[str, str, int, float]],
    top_k: int = 3,
    max_per_page: int = 2
) -> List[Tuple[str, str, int, float]]:
    """
    Rerank results to ensure diversity (different pages, different documents).
    
    Prevents returning too many chunks from the same page/document.
    
    Args:
        results: List of (chunk_text, filename, page_number, similarity) tuples
        top_k: Number of results to return
        max_per_page: Maximum chunks per (filename, page) combination
    
    Returns:
        Diversified results
    """
    selected = []
    page_counts: Dict[Tuple[str, int], int] = {}
    
    for result in results:
        chunk_text, filename, page_number, similarity = result
        page_key = (filename, page_number)
        
        # Count chunks from this page
        current_count = page_counts.get(page_key, 0)
        
        if current_count < max_per_page:
            selected.append(result)
            page_counts[page_key] = current_count + 1
            
            if len(selected) >= top_k:
                break
    
    return selected


def rerank_by_length_penalty(
    results: List[Tuple[str, str, int, float]],
    min_length: int = 50,
    max_length: int = 2000
) -> List[Tuple[str, str, int, float]]:
    """
    Filter results by chunk length and apply length-based scoring.
    
    Very short chunks might lack context, very long chunks might be noisy.
    
    Args:
        results: List of (chunk_text, filename, page_number, similarity) tuples
        min_length: Minimum chunk length in characters
        max_length: Maximum chunk length in characters
    
    Returns:
        Filtered and re-scored results
    """
    filtered = []
    
    for chunk_text, filename, page_number, similarity in results:
        chunk_len = len(chunk_text)
        
        # Filter by length
        if chunk_len < min_length or chunk_len > max_length:
            continue
        
        # Apply length-based penalty/bonus
        # Prefer chunks between 100-500 characters (optimal for context)
        if 100 <= chunk_len <= 500:
            length_score = 1.0
        elif chunk_len < 100:
            length_score = 0.8
        else:
            # Penalize very long chunks slightly
            length_score = max(0.9, 1.0 - (chunk_len - 500) / 10000)
        
        # Adjust similarity with length score
        adjusted_similarity = similarity * length_score
        filtered.append((chunk_text, filename, page_number, adjusted_similarity))
    
    # Re-sort by adjusted similarity
    filtered.sort(key=lambda x: x[3], reverse=True)
    return filtered


def rerank_chunks(
    query: str,
    results: List[Tuple[str, str, int, float]],
    strategy: str = "combined",
    top_k: int = 3,
    min_similarity: float = 0.7,
    **kwargs
) -> List[Tuple[str, str, int, float]]:
    """
    Main reranking function that applies multiple strategies.
    
    Args:
        query: Original query string
        results: List of (chunk_text, filename, page_number, similarity) tuples
        strategy: Reranking strategy ("threshold", "keyword", "diversity", "length", "combined")
        top_k: Number of results to return
        min_similarity: Minimum similarity threshold
        **kwargs: Additional strategy-specific parameters
    
    Returns:
        Reranked and filtered results
    """
    if not results:
        return []
    
    # Start with similarity threshold filtering
    filtered = rerank_by_similarity_threshold(results, min_similarity)
    
    if strategy == "threshold":
        return filtered[:top_k]
    
    elif strategy == "keyword":
        return rerank_by_keyword_overlap(query, filtered, top_k)
    
    elif strategy == "diversity":
        max_per_page = kwargs.get("max_per_page", 2)
        return rerank_by_diversity(filtered, top_k, max_per_page)
    
    elif strategy == "length":
        filtered = rerank_by_length_penalty(filtered)
        return filtered[:top_k]
    
    elif strategy == "combined":
        # Apply multiple strategies in sequence
        
        # 1. Length filtering
        filtered = rerank_by_length_penalty(filtered)
        
        # 2. Keyword reranking
        filtered = rerank_by_keyword_overlap(query, filtered, top_k * 2)  # Get more for diversity
        
        # 3. Diversity reranking
        max_per_page = kwargs.get("max_per_page", 2)
        filtered = rerank_by_diversity(filtered, top_k, max_per_page)
        
        return filtered
    
    else:
        # Default: just filter by threshold and return top_k
        return filtered[:top_k]


# Example usage:
if __name__ == "__main__":
    # Mock results
    mock_results = [
        ("This is about machine learning algorithms", "doc1.pdf", 1, 0.85),
        ("Machine learning is a subset of AI", "doc1.pdf", 1, 0.82),
        ("Deep learning uses neural networks", "doc2.pdf", 3, 0.75),
        ("AI systems can learn from data", "doc1.pdf", 2, 0.70),
    ]
    
    query = "What is machine learning?"
    
    # Test different strategies
    print("Original results:", mock_results)
    print("\nThreshold filtering (>0.7):")
    print(rerank_chunks(query, mock_results, strategy="threshold", top_k=3))
    print("\nKeyword reranking:")
    print(rerank_chunks(query, mock_results, strategy="keyword", top_k=3))
    print("\nCombined strategy:")
    print(rerank_chunks(query, mock_results, strategy="combined", top_k=3))

