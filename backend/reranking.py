"""
Reranking utilities for improving retrieval quality.

This module provides different reranking strategies to filter and reorder
retrieved chunks based on relevance to the query. Enhanced for mixed-content
documents with improved relevance scoring.
"""

from typing import List, Tuple, Dict, Any, Optional
import re
import hashlib
from config import RAGConfig


def rerank_by_similarity_threshold(
    results: List[Tuple[str, str, int, float]],
    min_similarity: Optional[float] = None
) -> List[Tuple[str, str, int, float]]:
    """
    Filter results by minimum similarity threshold.
    
    Args:
        results: List of (chunk_text, filename, page_number, similarity) tuples
        min_similarity: Minimum similarity score to include (0.0-1.0). If None, uses RAGConfig.MIN_SIMILARITY_THRESHOLD
    
    Returns:
        Filtered results above threshold
    """
    if min_similarity is None:
        min_similarity = RAGConfig.MIN_SIMILARITY_THRESHOLD
    return [r for r in results if r[3] >= min_similarity]


def rerank_by_keyword_overlap(
    query: str,
    results: List[Tuple[str, str, int, float]],
    top_k: int = 3
) -> List[Tuple[str, str, int, float]]:
    """
    Rerank results by keyword overlap with query.
    
    Enhanced version with:
    - TF-IDF-like weighting for important keywords
    - Position-based boosting (keywords at start of query are more important)
    - Phrase matching bonus
    
    Args:
        query: Original query string
        results: List of (chunk_text, filename, page_number, similarity) tuples
        top_k: Number of results to return
    
    Returns:
        Reranked results
    """
    # Extract keywords from query with position weighting
    query_tokens = re.findall(r'\b\w+\b', query.lower())
    query_words = set(query_tokens)
    
    # Weight keywords by position (earlier = more important)
    keyword_weights = {}
    for i, word in enumerate(query_tokens):
        position_weight = 1.0 + (len(query_tokens) - i) * 0.1
        keyword_weights[word] = keyword_weights.get(word, 0) + position_weight
    
    # Normalize weights
    max_weight = max(keyword_weights.values()) if keyword_weights else 1.0
    keyword_weights = {k: v / max_weight for k, v in keyword_weights.items()}
    
    scored_results = []
    for chunk_text, filename, page_number, similarity in results:
        chunk_lower = chunk_text.lower()
        chunk_words = set(re.findall(r'\b\w+\b', chunk_lower))
        
        # Calculate weighted keyword overlap
        weighted_overlap = sum(
            keyword_weights.get(word, 0.5) 
            for word in query_words & chunk_words
        )
        keyword_score = weighted_overlap / len(query_words) if query_words else 0
        
        # Phrase matching bonus (exact phrase matches get boost)
        phrase_bonus = 0.0
        if len(query_tokens) >= 2:
            # Check for 2-3 word phrases
            for i in range(len(query_tokens) - 1):
                phrase = ' '.join(query_tokens[i:i+2])
                if phrase in chunk_lower:
                    phrase_bonus += 0.1
                if i < len(query_tokens) - 2:
                    phrase3 = ' '.join(query_tokens[i:i+3])
                    if phrase3 in chunk_lower:
                        phrase_bonus += 0.15
        
        keyword_score = min(1.0, keyword_score + phrase_bonus)
        
        # Combine similarity score with keyword score (weighted)
        combined_score = (similarity * RAGConfig.SIMILARITY_WEIGHT) + (keyword_score * RAGConfig.KEYWORD_WEIGHT)
        
        scored_results.append((chunk_text, filename, page_number, combined_score))
    
    # Sort by combined score and return top_k
    scored_results.sort(key=lambda x: x[3], reverse=True)
    return scored_results[:top_k]


def detect_content_type(chunk_text: str) -> str:
    """
    Detect the type of content in a chunk (text, list, table, code, etc.).
    
    Args:
        chunk_text: Chunk text to analyze
    
    Returns:
        Content type: "text", "list", "table", "code", "mixed"
    """
    text_lower = chunk_text.lower()
    
    # Check for code-like content
    code_indicators = ['def ', 'function', 'import ', 'class ', '{', '}', '()', '=>']
    if any(indicator in text_lower for indicator in code_indicators):
        return "code"
    
    # Check for list-like content
    lines = chunk_text.strip().split('\n')
    list_indicators = 0
    for line in lines[:5]:  # Check first 5 lines
        stripped = line.strip()
        if stripped.startswith(('-', '*', '•', '1.', '2.', '3.')):
            list_indicators += 1
    
    if list_indicators >= 2:
        return "list"
    
    # Check for table-like content (multiple tabs or consistent spacing)
    tab_count = sum(1 for line in lines[:5] if '\t' in line)
    if tab_count >= 3:
        return "table"
    
    # Check for mixed content
    if list_indicators > 0 and len(lines) > 3:
        return "mixed"
    
    return "text"


def rerank_by_content_relevance(
    query: str,
    results: List[Tuple[str, str, int, float]],
    top_k: int = 3
) -> List[Tuple[str, str, int, float]]:
    """
    Rerank results by content type relevance for mixed-content documents.
    
    Enhanced version with:
    - Better query type detection
    - Semantic structure analysis
    - Answer completeness scoring
    
    Args:
        query: Original query string
        results: List of (chunk_text, filename, page_number, similarity) tuples
        top_k: Number of results to return
    
    Returns:
        Reranked results with content-aware scoring
    """
    query_lower = query.lower()
    
    # Enhanced query type detection
    is_definition_query = any(phrase in query_lower for phrase in [
        'what is', 'what are', 'define', 'explain', 'meaning', 'definition',
        'what does', 'what do', 'describe'
    ])
    is_list_query = any(phrase in query_lower for phrase in [
        'list', 'examples', 'types of', 'kinds of', 'categories', 'varieties',
        'what are the', 'name the', 'enumerate'
    ])
    is_howto_query = any(phrase in query_lower for phrase in [
        'how', 'steps', 'process', 'method', 'procedure', 'way to',
        'how do you', 'how can', 'how to'
    ])
    is_comparison_query = any(phrase in query_lower for phrase in [
        'compare', 'difference', 'versus', 'vs', 'better', 'worse',
        'advantages', 'disadvantages'
    ])
    
    scored_results = []
    for chunk_text, filename, page_number, similarity in results:
        content_type = detect_content_type(chunk_text)
        content_score = 1.0
        
        # Enhanced content type matching
        if is_list_query and content_type in ["list", "table"]:
            content_score = 1.25
        elif is_definition_query and content_type == "text":
            content_score = 1.2
        elif is_howto_query and content_type in ["list", "mixed"]:
            content_score = 1.15
        elif is_comparison_query and content_type in ["text", "mixed"]:
            content_score = 1.1
        
        # Penalize very short chunks (likely incomplete)
        chunk_len = len(chunk_text.strip())
        if chunk_len < 30:
            content_score *= 0.6
        elif chunk_len < 50:
            content_score *= 0.85
        
        # Boost chunks with good structure (paragraphs, complete sentences)
        sentence_count = chunk_text.count('.') + chunk_text.count('!') + chunk_text.count('?')
        if sentence_count >= 3:
            content_score *= 1.1
        elif sentence_count >= 2:
            content_score *= 1.05
        
        # Boost chunks that start with capital letters (likely complete sentences)
        if chunk_text.strip() and chunk_text.strip()[0].isupper():
            content_score *= 1.05
        
        # Penalize chunks that are mostly punctuation or whitespace
        alphanumeric_ratio = sum(1 for c in chunk_text if c.isalnum()) / len(chunk_text) if chunk_text else 0
        if alphanumeric_ratio < 0.5:
            content_score *= 0.8
        
        # Combine similarity with content score
        final_score = similarity * content_score
        scored_results.append((chunk_text, filename, page_number, final_score))
    
    # Sort by final score
    scored_results.sort(key=lambda x: x[3], reverse=True)
    return scored_results[:top_k]


def rerank_by_diversity(
    results: List[Tuple[str, str, int, float]],
    top_k: int = 3,
    max_per_page: Optional[int] = None,
    max_per_document: Optional[int] = None
) -> List[Tuple[str, str, int, float]]:
    """
    Rerank results to ensure diversity (different pages, different documents).
    
    Prevents returning too many chunks from the same page/document.
    Enhanced for mixed-content documents.
    
    Args:
        results: List of (chunk_text, filename, page_number, similarity) tuples
        top_k: Number of results to return
        max_per_page: Maximum chunks per (filename, page) combination. If None, uses RAGConfig.MAX_CHUNKS_PER_PAGE
        max_per_document: Maximum chunks per document (None = no limit, uses RAGConfig.MAX_CHUNKS_PER_DOCUMENT if set)
    
    Returns:
        Diversified results
    """
    if max_per_page is None:
        max_per_page = RAGConfig.MAX_CHUNKS_PER_PAGE
    if max_per_document is None:
        max_per_document = RAGConfig.MAX_CHUNKS_PER_DOCUMENT
    
    selected = []
    page_counts: Dict[Tuple[str, int], int] = {}
    doc_counts: Dict[str, int] = {}
    
    for result in results:
        chunk_text, filename, page_number, similarity = result
        page_key = (filename, page_number)
        
        # Count chunks from this page
        current_page_count = page_counts.get(page_key, 0)
        
        # Count chunks from this document
        current_doc_count = doc_counts.get(filename, 0)
        
        # Check limits
        page_limit_ok = current_page_count < max_per_page
        doc_limit_ok = max_per_document is None or current_doc_count < max_per_document
        
        if page_limit_ok and doc_limit_ok:
            selected.append(result)
            page_counts[page_key] = current_page_count + 1
            doc_counts[filename] = current_doc_count + 1
            
            if len(selected) >= top_k:
                break
    
    return selected


def rerank_by_length_penalty(
    results: List[Tuple[str, str, int, float]],
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> List[Tuple[str, str, int, float]]:
    """
    Filter results by chunk length and apply length-based scoring.
    
    Very short chunks might lack context, very long chunks might be noisy.
    
    Args:
        results: List of (chunk_text, filename, page_number, similarity) tuples
        min_length: Minimum chunk length in characters. If None, uses RAGConfig.MIN_CHUNK_LENGTH
        max_length: Maximum chunk length in characters. If None, uses RAGConfig.MAX_CHUNK_LENGTH
    
    Returns:
        Filtered and re-scored results
    """
    if min_length is None:
        min_length = RAGConfig.MIN_CHUNK_LENGTH
    if max_length is None:
        max_length = RAGConfig.MAX_CHUNK_LENGTH
    
    filtered = []
    
    for chunk_text, filename, page_number, similarity in results:
        chunk_len = len(chunk_text)
        
        # Filter by length
        if chunk_len < min_length or chunk_len > max_length:
            continue
        
        # Apply length-based penalty/bonus
        # Prefer chunks in optimal range
        if RAGConfig.OPTIMAL_CHUNK_LENGTH_MIN <= chunk_len <= RAGConfig.OPTIMAL_CHUNK_LENGTH_MAX:
            length_score = 1.0
        elif chunk_len < RAGConfig.OPTIMAL_CHUNK_LENGTH_MIN:
            length_score = 0.8
        else:
            # Penalize very long chunks slightly
            length_score = max(0.9, 1.0 - (chunk_len - RAGConfig.OPTIMAL_CHUNK_LENGTH_MAX) / 10000)
        
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
        max_per_page = kwargs.get("max_per_page", RAGConfig.MAX_CHUNKS_PER_PAGE)
        max_per_document = kwargs.get("max_per_document", RAGConfig.MAX_CHUNKS_PER_DOCUMENT)
        return rerank_by_diversity(filtered, top_k, max_per_page, max_per_document)
    
    elif strategy == "length":
        filtered = rerank_by_length_penalty(filtered)
        return filtered[:top_k]
    
    elif strategy == "combined":
        # Apply multiple strategies in sequence for mixed-content documents
        # Enhanced pipeline for higher precision
        
        # 1. Length filtering (remove very short/long chunks early)
        filtered = rerank_by_length_penalty(filtered)
        
        # 2. Content-aware reranking (for mixed-content documents)
        # Keep more candidates for next stages
        filtered = rerank_by_content_relevance(query, filtered, top_k * 4)
        
        # 3. Keyword reranking with enhanced scoring
        filtered = rerank_by_keyword_overlap(query, filtered, top_k * 3)
        
        # 4. Diversity reranking (enhanced) - ensures we get diverse sources
        max_per_page = kwargs.get("max_per_page", RAGConfig.MAX_CHUNKS_PER_PAGE)
        max_per_document = kwargs.get("max_per_document", RAGConfig.MAX_CHUNKS_PER_DOCUMENT)
        filtered = rerank_by_diversity(filtered, top_k, max_per_page, max_per_document)
        
        # 5. Final similarity boost - ensure top results have high similarity
        # Re-sort by similarity as tie-breaker for final ranking
        filtered.sort(key=lambda x: x[3], reverse=True)
        
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

