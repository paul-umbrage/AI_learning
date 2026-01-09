"""
Query expansion for improved retrieval.

Generates multiple query variations to improve search coverage.
"""

from typing import List, Dict, Any, Optional, Set
from openai import OpenAI
import os
from dotenv import load_dotenv
from utils.error_handling import retry_with_backoff

load_dotenv()


@retry_with_backoff(max_retries=2, initial_delay=1.0)
def expand_query(
    query: str,
    num_variations: int = 3,
    client: Optional[OpenAI] = None,
    model: str = "gpt-3.5-turbo"
) -> List[str]:
    """
    Generate query variations using LLM.
    
    Args:
        query: Original query
        num_variations: Number of variations to generate
        client: Optional OpenAI client
        model: Model to use for expansion
    
    Returns:
        List of query variations (includes original)
    """
    if not client:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return [query]  # Return original if no API key
        client = OpenAI(api_key=api_key)
    
    try:
        expansion_prompt = f"""Generate {num_variations} different ways to ask the following question. 
Each variation should:
- Use different wording
- Focus on different aspects if possible
- Maintain the same core intent
- Be concise (1-2 sentences max)

Original question: {query}

Generate {num_variations} variations, one per line, numbered 1-{num_variations}:"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates query variations for search."},
                {"role": "user", "content": expansion_prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        variations_text = response.choices[0].message.content.strip()
        
        # Parse variations
        variations = []
        for line in variations_text.split('\n'):
            line = line.strip()
            # Remove numbering (e.g., "1. ", "1) ", "- ")
            line = line.lstrip('0123456789.-) ')
            if line and len(line) > 10:  # Filter out very short lines
                variations.append(line)
        
        # Add original query
        all_queries = [query] + variations[:num_variations]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in all_queries:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)
        
        return unique_queries[:num_variations + 1]  # Return original + variations
    
    except Exception as e:
        # If expansion fails, return original query
        return [query]


def merge_search_results(
    all_results: List[List[tuple]],
    top_k: int = 5,
    deduplicate: bool = True
) -> List[tuple]:
    """
    Merge results from multiple queries and remove duplicates.
    
    Args:
        all_results: List of result lists from different queries
        top_k: Number of final results to return
        deduplicate: Whether to remove duplicate chunks
    
    Returns:
        Merged and deduplicated results: (chunk_text, filename, page_number, score)
    """
    # Combine all results
    combined = []
    seen_chunks = set() if deduplicate else None
    
    for results in all_results:
        for result in results:
            chunk_text, filename, page_number, score = result
            
            # Create unique identifier for deduplication
            if deduplicate:
                chunk_id = (chunk_text[:100], filename, page_number)  # Use first 100 chars as ID
                if chunk_id in seen_chunks:
                    continue
                seen_chunks.add(chunk_id)
            
            combined.append((chunk_text, filename, page_number, score))
    
    # Sort by score (descending)
    combined.sort(key=lambda x: x[3], reverse=True)
    
    # Return top_k
    return combined[:top_k]


def expand_and_search(
    query: str,
    search_func: callable,
    num_variations: int = 2,
    top_k: int = 5,
    client: Optional[OpenAI] = None
) -> List[tuple]:
    """
    Expand query and search with all variations.
    
    Args:
        query: Original query
        search_func: Function that takes (query, top_k) and returns results
        num_variations: Number of query variations to generate
        top_k: Number of final results to return
        client: Optional OpenAI client
    
    Returns:
        Merged search results
    """
    # Generate query variations
    queries = expand_query(query, num_variations=num_variations, client=client)
    
    # Search with each query
    all_results = []
    for q in queries:
        try:
            results = search_func(q, top_k=top_k * 2)  # Get more results per query
            all_results.append(results)
        except Exception:
            # If search fails for a variation, continue with others
            continue
    
    # Merge and deduplicate results
    merged_results = merge_search_results(all_results, top_k=top_k, deduplicate=True)
    
    return merged_results


def simple_query_expansion(query: str) -> List[str]:
    """
    Simple query expansion without LLM (rule-based).
    
    Useful as fallback when LLM is unavailable.
    
    Args:
        query: Original query
    
    Returns:
        List of query variations
    """
    variations = [query]  # Always include original
    
    # Add variations with different question words
    question_words = {
        'what': ['what is', 'what are', 'explain', 'describe'],
        'how': ['how does', 'how do', 'how is', 'how are'],
        'why': ['why is', 'why are', 'what causes', 'what leads to'],
        'when': ['when does', 'when do', 'at what time'],
        'where': ['where is', 'where are', 'in what location']
    }
    
    query_lower = query.lower()
    for word, alternatives in question_words.items():
        if query_lower.startswith(word):
            for alt in alternatives:
                if alt != word:
                    variation = query.replace(word, alt, 1)
                    if variation not in variations:
                        variations.append(variation)
            break
    
    # Add variations with synonyms for common terms
    synonyms = {
        'is': ['are', 'means', 'refers to'],
        'does': ['do', 'performs', 'executes'],
        'can': ['could', 'is able to', 'has the ability to']
    }
    
    for term, syns in synonyms.items():
        if f' {term} ' in f' {query_lower} ':
            for syn in syns:
                variation = query.replace(f' {term} ', f' {syn} ', 1)
                if variation not in variations:
                    variations.append(variation)
    
    return variations[:4]  # Limit to 4 variations

