"""
Context assembly utilities for building RAG context from retrieved chunks.

Provides deduplication, smart context window management, and token-aware
truncation to optimize context quality and stay within model limits.
"""

from typing import List, Tuple, Dict, Any, Optional
import hashlib
import tiktoken


def calculate_chunk_hash(chunk_text: str, filename: str, page_number: int) -> str:
    """
    Generate a hash for a chunk to detect duplicates.
    
    Uses normalized text (first 200 chars) + filename + page for deduplication.
    
    Args:
        chunk_text: Chunk text
        filename: Document filename
        page_number: Page number
    
    Returns:
        Hash string for deduplication
    """
    # Normalize: lowercase, strip whitespace, take first 200 chars
    normalized = chunk_text.lower().strip()[:200]
    content = f"{normalized}|{filename}|{page_number}"
    return hashlib.md5(content.encode()).hexdigest()


def deduplicate_chunks(
    results: List[Tuple[str, str, int, float]],
    similarity_threshold: float = 0.85
) -> List[Tuple[str, str, int, float]]:
    """
    Remove duplicate or highly similar chunks from results.
    
    Uses both hash-based exact deduplication and similarity-based
    near-duplicate detection.
    
    Args:
        results: List of (chunk_text, filename, page_number, similarity) tuples
        similarity_threshold: Minimum similarity to consider chunks duplicates
    
    Returns:
        Deduplicated results
    """
    if not results:
        return []
    
    seen_hashes = set()
    seen_chunks = []
    deduplicated = []
    
    for chunk_text, filename, page_number, similarity in results:
        # Hash-based exact deduplication
        chunk_hash = calculate_chunk_hash(chunk_text, filename, page_number)
        if chunk_hash in seen_hashes:
            continue
        
        # Similarity-based near-duplicate detection
        is_duplicate = False
        chunk_words = set(chunk_text.lower().split())
        
        for seen_text, _, _, _ in seen_chunks:
            seen_words = set(seen_text.lower().split())
            
            # Calculate word overlap
            if len(chunk_words) == 0 or len(seen_words) == 0:
                continue
            
            overlap = len(chunk_words & seen_words)
            union = len(chunk_words | seen_words)
            jaccard_similarity = overlap / union if union > 0 else 0
            
            # If very similar, consider it a duplicate
            if jaccard_similarity >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            deduplicated.append((chunk_text, filename, page_number, similarity))
            seen_hashes.add(chunk_hash)
            seen_chunks.append((chunk_text, filename, page_number, similarity))
    
    return deduplicated


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        model: Model name for encoding
    
    Returns:
        Number of tokens
    """
    try:
        # Map model names to encodings
        encoding_name = {
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-4": "cl100k_base",
            "gpt-4-turbo-preview": "cl100k_base",
            "gpt-4o": "cl100k_base",
            "gpt-4o-mini": "cl100k_base"
        }.get(model, "cl100k_base")
        
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4


def assemble_context(
    results: List[Tuple[str, str, int, float]],
    max_tokens: int = 2000,
    model: str = "gpt-3.5-turbo",
    deduplicate: bool = True,
    prioritize_high_similarity: bool = True
) -> Tuple[str, List[Dict[str, Any]], int]:
    """
    Assemble context from retrieved chunks with smart window management.
    
    Features:
    - Deduplication of similar chunks
    - Token-aware truncation
    - Priority-based selection (high similarity first)
    - Smart truncation of individual chunks if needed
    
    Args:
        results: List of (chunk_text, filename, page_number, similarity) tuples
        max_tokens: Maximum tokens for context window
        model: Model name for token counting
        deduplicate: Whether to deduplicate chunks
        prioritize_high_similarity: Prioritize high-similarity chunks
    
    Returns:
        Tuple of (context_string, sources_list, tokens_used)
    """
    if not results:
        return "", [], 0
    
    # Deduplicate if enabled
    if deduplicate:
        results = deduplicate_chunks(results)
    
    # Sort by similarity if prioritizing
    if prioritize_high_similarity:
        results = sorted(results, key=lambda x: x[3], reverse=True)
    
    # Build context with token management
    context_parts = []
    sources = []
    total_tokens = 0
    header_tokens = count_tokens("[Context X]\n", model)  # Approximate header tokens
    
    for i, (chunk_text, filename, page_number, similarity) in enumerate(results, 1):
        # Estimate tokens for this chunk with header
        chunk_tokens = count_tokens(chunk_text, model) + header_tokens
        
        # If adding this chunk would exceed limit, try truncating it
        if total_tokens + chunk_tokens > max_tokens:
            # Calculate available tokens
            available_tokens = max_tokens - total_tokens - header_tokens
            
            if available_tokens > 50:  # Only include if we have meaningful space
                # Truncate chunk to fit
                # Rough estimate: 1 token ≈ 4 chars, but be conservative
                max_chars = available_tokens * 3  # Conservative estimate
                truncated_chunk = chunk_text[:max_chars]
                
                # Try to truncate at sentence boundary
                last_period = truncated_chunk.rfind('.')
                last_newline = truncated_chunk.rfind('\n')
                truncate_at = max(last_period, last_newline)
                
                if truncate_at > max_chars * 0.7:  # Only if we keep most of it
                    truncated_chunk = truncated_chunk[:truncate_at + 1]
                
                chunk_text = truncated_chunk + "..."
                chunk_tokens = count_tokens(chunk_text, model) + header_tokens
            else:
                # Not enough space, stop adding chunks
                break
        
        # Add chunk to context
        context_parts.append(f"[Context {i}]\n{chunk_text}\n")
        sources.append({
            "chunk_index": i,
            "filename": filename,
            "page_number": int(page_number),
            "similarity": float(similarity),
            "text_preview": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
        })
        
        total_tokens += chunk_tokens
        
        # Stop if we've reached the limit
        if total_tokens >= max_tokens:
            break
    
    context = "\n".join(context_parts)
    
    # Final token count (more accurate)
    final_tokens = count_tokens(context, model)
    
    return context, sources, final_tokens


def optimize_context_order(
    results: List[Tuple[str, str, int, float]],
    query: str
) -> List[Tuple[str, str, int, float]]:
    """
    Optimize the order of chunks for better context flow.
    
    Tries to:
    - Group chunks from same document/page together
    - Put highest similarity chunks first
    - Ensure logical flow (earlier pages before later pages)
    
    Args:
        results: List of (chunk_text, filename, page_number, similarity) tuples
        query: Original query for context
    
    Returns:
        Reordered results
    """
    if not results:
        return []
    
    # Sort by: filename, then page number, then similarity (descending)
    # This groups related chunks together while maintaining quality
    sorted_results = sorted(
        results,
        key=lambda x: (x[1], x[2], -x[3])  # filename, page, -similarity (desc)
    )
    
    # But prioritize high similarity overall
    # Re-sort to put highest similarity chunks first, but keep grouping
    high_sim = [r for r in sorted_results if r[3] >= 0.8]
    medium_sim = [r for r in sorted_results if 0.7 <= r[3] < 0.8]
    low_sim = [r for r in sorted_results if r[3] < 0.7]
    
    # Combine: high first, then medium, then low
    optimized = high_sim + medium_sim + low_sim
    
    return optimized
