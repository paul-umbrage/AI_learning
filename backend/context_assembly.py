"""
Context assembly utilities for building RAG context from retrieved chunks.

Provides deduplication, smart context window management, and token-aware
truncation to optimize context quality and stay within model limits.
"""

from typing import List, Tuple, Dict, Any, Optional
import hashlib
import tiktoken
from config import RAGConfig


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
    similarity_threshold: Optional[float] = None
) -> List[Tuple[str, str, int, float]]:
    """
    Remove duplicate or highly similar chunks from results.
    
    Uses both hash-based exact deduplication and similarity-based
    near-duplicate detection.
    
    Args:
        results: List of (chunk_text, filename, page_number, similarity) tuples
        similarity_threshold: Minimum similarity to consider chunks duplicates. If None, uses RAGConfig.DEDUPLICATION_SIMILARITY_THRESHOLD
    
    Returns:
        Deduplicated results
    """
    if similarity_threshold is None:
        similarity_threshold = RAGConfig.DEDUPLICATION_SIMILARITY_THRESHOLD
    
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
    max_tokens: Optional[int] = None,
    model: str = "gpt-3.5-turbo",
    deduplicate: bool = True,
    prioritize_high_similarity: bool = True,
    reserve_tokens: Optional[int] = None
) -> Tuple[str, List[Dict[str, Any]], int]:
    """
    Assemble context from retrieved chunks with smart window management.
    
    Enhanced features:
    - Deduplication of similar chunks
    - Token-aware truncation with better boundary detection
    - Priority-based selection (high similarity first)
    - Smart truncation of individual chunks at sentence/paragraph boundaries
    - Token reservation for prompt overhead
    - Better token estimation
    
    Args:
        results: List of (chunk_text, filename, page_number, similarity) tuples
        max_tokens: Maximum tokens for context window
        model: Model name for token counting
        deduplicate: Whether to deduplicate chunks
        prioritize_high_similarity: Prioritize high-similarity chunks
        reserve_tokens: Tokens to reserve for prompt overhead (default: 200)
    
    Returns:
        Tuple of (context_string, sources_list, tokens_used)
    """
    if not results:
        return "", [], 0
    
    # Ensure max_tokens and reserve_tokens have defaults
    if max_tokens is None:
        max_tokens = 2000
    if reserve_tokens is None:
        reserve_tokens = 200
    
    # Reserve tokens for prompt overhead
    effective_max_tokens = max(100, max_tokens - reserve_tokens)
    
    # Deduplicate if enabled
    if deduplicate:
        results = deduplicate_chunks(results)
    
    # Sort by similarity if prioritizing
    if prioritize_high_similarity:
        results = sorted(results, key=lambda x: x[3], reverse=True)
    
    # Build context with enhanced token management
    context_parts = []
    sources = []
    total_tokens = 0
    header_tokens = count_tokens("[Context X]\n", model)  # Approximate header tokens
    
    # Debug: Log input
    import logging
    logger = logging.getLogger("rag_system")
    logger.debug(f"assemble_context: received {len(results)} results")
    
    for i, result in enumerate(results, 1):
        # Handle both tuple and list formats - database returns tuples
        try:
            if isinstance(result, (list, tuple)):
                if len(result) >= 4:
                    chunk_text, filename, page_number, similarity = result[0], result[1], result[2], result[3]
                else:
                    logger.error(f"Result has insufficient elements at index {i-1}: {result}, length: {len(result)}")
                    continue
            else:
                logger.error(f"Invalid result type at index {i-1}: {type(result)}, value: {result}")
                continue
        except Exception as e:
            logger.error(f"Error unpacking result at index {i-1}: {e}, result: {result}")
            continue
        
        # CRITICAL: Ensure we have valid values
        if not chunk_text or not isinstance(chunk_text, str):
            logger.error(f"Invalid chunk_text at index {i-1}: {type(chunk_text)}, value: {str(chunk_text)[:100]}")
            continue
        
        # Estimate tokens for this chunk with header
        chunk_tokens = count_tokens(chunk_text, model)
        chunk_with_header_tokens = chunk_tokens + header_tokens
        
        # If adding this chunk would exceed limit, try truncating it
        if total_tokens + chunk_with_header_tokens > effective_max_tokens:
            # Calculate available tokens
            available_tokens = effective_max_tokens - total_tokens - header_tokens
            
            if available_tokens > 30:  # Only include if we have meaningful space (lowered threshold)
                # Better truncation: try to preserve complete sentences
                # Estimate characters per token (conservative: 3 chars per token)
                max_chars = available_tokens * 3
                truncated_chunk = chunk_text[:max_chars]
                
                # Try to truncate at sentence boundary (prefer periods)
                best_truncate = -1
                
                # Look for sentence endings in reverse order
                for delimiter in ['. ', '.\n', '! ', '?\n', '?\n']:
                    pos = truncated_chunk.rfind(delimiter)
                    if pos > max_chars * 0.5:  # Only if we keep at least 50%
                        best_truncate = pos + len(delimiter)
                        break
                
                # Fallback to paragraph boundary
                if best_truncate == -1:
                    best_truncate = truncated_chunk.rfind('\n\n')
                    if best_truncate > max_chars * 0.5:
                        best_truncate += 2
                
                # Fallback to single newline
                if best_truncate == -1:
                    best_truncate = truncated_chunk.rfind('\n')
                    if best_truncate > max_chars * 0.6:
                        best_truncate += 1
                
                if best_truncate > max_chars * 0.5:
                    truncated_chunk = truncated_chunk[:best_truncate]
                else:
                    # Last resort: hard truncate but add ellipsis
                    truncated_chunk = truncated_chunk[:max_chars - 3]
                
                chunk_text = truncated_chunk + ("..." if len(truncated_chunk) < len(chunk_text) else "")
                chunk_tokens = count_tokens(chunk_text, model)
                chunk_with_header_tokens = chunk_tokens + header_tokens
            else:
                # Not enough space, stop adding chunks
                break
        
        # Add chunk to context
        context_parts.append(f"[Context {i}]\n{chunk_text}\n")
        
        # Build source entry - ensure all fields are properly set
        # CRITICAL: Build source BEFORE adding to context_parts to ensure they stay in sync
        try:
            source_entry = {
                "chunk_index": i,
                "filename": str(filename) if filename else "unknown",
                "page_number": int(page_number) if page_number is not None else 0,
                "similarity": float(similarity) if similarity is not None else 0.0,
                "text_preview": (chunk_text[:100] + "...") if len(chunk_text) > 100 else chunk_text,
            }
            
            # Add optional fields
            try:
                source_entry["tokens"] = chunk_tokens
                if i <= len(results):
                    original_chunk = results[i-1][0] if isinstance(results[i-1], (list, tuple)) and len(results[i-1]) > 0 else chunk_text
                    source_entry["truncated"] = len(chunk_text) < len(original_chunk)
                else:
                    source_entry["truncated"] = False
            except Exception as e:
                logger.warning(f"Error adding optional source fields: {e}")
                source_entry["truncated"] = False
            
            # CRITICAL: Always append source immediately after building context
            sources.append(source_entry)
            logger.debug(f"Added source {i}: filename={source_entry['filename']}, page={source_entry['page_number']}")
            
        except Exception as e:
            logger.error(f"CRITICAL: Error building source entry at index {i}: {e}, result: {result}")
            # Create minimal source entry as fallback - MUST have a source for every context part
            sources.append({
                "chunk_index": i,
                "filename": str(filename) if filename else "unknown",
                "page_number": int(page_number) if page_number is not None else 0,
                "similarity": float(similarity) if similarity is not None else 0.0,
                "text_preview": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
            })
        
        total_tokens += chunk_with_header_tokens
        
        # Stop if we've reached the limit
        if total_tokens >= effective_max_tokens:
            break
    
    context = "\n".join(context_parts)
    
    # Final token count (more accurate)
    final_tokens = count_tokens(context, model)
    
    # CRITICAL: Ensure sources list matches context_parts
    # If we built context but sources are empty, something went wrong
    if len(context_parts) > 0 and len(sources) == 0:
        logger.error(
            f"assemble_context: CRITICAL - context built ({len(context_parts)} parts) but sources empty! "
            f"results_count={len(results)}, context_length={len(context)}, "
            f"results_type={type(results)}, first_result_type={type(results[0]) if results else 'None'}"
        )
        # Try to rebuild sources from results if we still have them
        if results and len(results) > 0:
            logger.warning("Attempting to rebuild sources from results...")
            for i, result in enumerate(results[:len(context_parts)], 1):
                try:
                    if isinstance(result, (list, tuple)) and len(result) >= 4:
                        chunk_text, filename, page_number, similarity = result[0], result[1], result[2], result[3]
                        sources.append({
                            "chunk_index": i,
                            "filename": str(filename) if filename else "unknown",
                            "page_number": int(page_number) if page_number is not None else 0,
                            "similarity": float(similarity) if similarity is not None else 0.0,
                            "text_preview": (chunk_text[:100] + "...") if len(chunk_text) > 100 else chunk_text
                        })
                    else:
                        # Fallback if result format is wrong
                        sources.append({
                            "chunk_index": i,
                            "filename": "unknown",
                            "page_number": 0,
                            "similarity": 0.0,
                            "text_preview": context_parts[i-1][:100] + "..." if i <= len(context_parts) else ""
                        })
                except Exception as e:
                    logger.error(f"Error rebuilding source {i}: {e}")
                    sources.append({
                        "chunk_index": i,
                        "filename": "unknown",
                        "page_number": 0,
                        "similarity": 0.0,
                        "text_preview": context_parts[i-1][:100] + "..." if i <= len(context_parts) else ""
                    })
            logger.warning(f"assemble_context: Rebuilt {len(sources)} sources from results as fallback")
        else:
            # Last resort: rebuild from context_parts (loses filename/page info)
            for i, part in enumerate(context_parts, 1):
                sources.append({
                    "chunk_index": i,
                    "filename": "unknown",
                    "page_number": 0,
                    "similarity": 0.0,
                    "text_preview": part[:100] + "..." if len(part) > 100 else part
                })
            logger.warning(f"assemble_context: Rebuilt {len(sources)} sources from context_parts as last resort")
    
    # Debug: Log output
    logger.debug(f"assemble_context: returning context_length={len(context)}, sources_count={len(sources)}, tokens={final_tokens}")
    
    # Ensure sources is always a list
    if not isinstance(sources, list):
        logger.error(f"assemble_context: sources is not a list! Type: {type(sources)}, Value: {sources}")
        sources = []
    
    # Final validation: sources count should match context parts
    if len(sources) != len(context_parts):
        logger.warning(
            f"assemble_context: Mismatch! context_parts={len(context_parts)}, sources={len(sources)}"
        )
    
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
