"""
Hallucination detection for RAG responses.

Detects when LLM responses contain information not present in
the retrieved context.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


def extract_claims(text: str) -> List[str]:
    """
    Extract factual claims from text.
    
    Simple heuristic: extract sentences that make factual statements.
    
    Args:
        text: Text to extract claims from
    
    Returns:
        List of claim strings
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Filter out very short sentences and questions
    claims = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20 and not sentence.endswith('?'):
            # Remove common non-factual phrases
            if not any(phrase in sentence.lower() for phrase in [
                'i think', 'i believe', 'in my opinion', 'perhaps', 'maybe',
                'might be', 'could be', 'possibly'
            ]):
                claims.append(sentence)
    
    return claims


def calculate_semantic_similarity(text1: str, text2: str, client: Optional[OpenAI] = None) -> float:
    """
    Calculate semantic similarity between two texts using embeddings.
    
    Args:
        text1: First text
        text2: Second text
        client: Optional OpenAI client (creates new one if not provided)
    
    Returns:
        Similarity score (0.0-1.0)
    """
    if not client:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return 0.0
        client = OpenAI(api_key=api_key)
    
    try:
        # Get embeddings
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[text1, text2]
        )
        
        embedding1 = response.data[0].embedding
        embedding2 = response.data[1].embedding
        
        # Calculate cosine similarity
        import numpy as np
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    except Exception:
        return 0.0


def check_claim_in_context(claim: str, context: str, threshold: float = 0.7) -> Tuple[bool, float]:
    """
    Check if a claim is supported by the context.
    
    Args:
        claim: Claim to verify
        context: Context to check against
        threshold: Similarity threshold (default: 0.7)
    
    Returns:
        Tuple of (is_supported, similarity_score)
    """
    # Simple keyword matching first (fast)
    claim_lower = claim.lower()
    context_lower = context.lower()
    
    # Extract key terms from claim
    claim_terms = set(re.findall(r'\b\w+\b', claim_lower))
    # Remove common stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been'}
    claim_terms = {t for t in claim_terms if len(t) > 3 and t not in stopwords}
    
    if not claim_terms:
        return False, 0.0
    
    # Check if key terms appear in context
    context_terms = set(re.findall(r'\b\w+\b', context_lower))
    overlap = len(claim_terms & context_terms) / len(claim_terms) if claim_terms else 0.0
    
    # If good keyword overlap, likely supported
    if overlap > 0.5:
        return True, min(1.0, overlap)
    
    # Otherwise, use semantic similarity
    similarity = calculate_semantic_similarity(claim, context)
    is_supported = similarity >= threshold
    
    return is_supported, similarity


def detect_hallucinations(
    response: str,
    context: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    threshold: float = 0.7,
    use_llm_verification: bool = False,
    client: Optional[OpenAI] = None
) -> Dict[str, Any]:
    """
    Detect hallucinations in LLM response.
    
    Args:
        response: LLM-generated response
        context: Retrieved context from RAG
        sources: Optional list of source chunks
        threshold: Similarity threshold for claim verification
        use_llm_verification: Use LLM to verify claims (more accurate but slower)
        client: Optional OpenAI client
    
    Returns:
        Dictionary with:
        - has_hallucinations: bool
        - hallucination_score: float (0.0-1.0, higher = more hallucinations)
        - unsupported_claims: List[str]
        - supported_claims: List[str]
        - verification_details: List[Dict]
    """
    if not context or not response:
        return {
            'has_hallucinations': False,
            'hallucination_score': 0.0,
            'unsupported_claims': [],
            'supported_claims': [],
            'verification_details': []
        }
    
    # Extract claims from response
    claims = extract_claims(response)
    
    if not claims:
        return {
            'has_hallucinations': False,
            'hallucination_score': 0.0,
            'unsupported_claims': [],
            'supported_claims': [],
            'verification_details': []
        }
    
    # Verify each claim
    unsupported_claims = []
    supported_claims = []
    verification_details = []
    
    for claim in claims:
        is_supported, similarity = check_claim_in_context(claim, context, threshold)
        
        verification_details.append({
            'claim': claim,
            'is_supported': is_supported,
            'similarity': similarity
        })
        
        if is_supported:
            supported_claims.append(claim)
        else:
            unsupported_claims.append(claim)
    
    # Calculate hallucination score
    total_claims = len(claims)
    hallucination_score = len(unsupported_claims) / total_claims if total_claims > 0 else 0.0
    
    # Use LLM verification if requested (more accurate)
    if use_llm_verification and unsupported_claims and client:
        # Verify unsupported claims with LLM
        verified_unsupported = []
        for claim in unsupported_claims:
            verification_prompt = f"""Is the following claim supported by the context below?

Claim: {claim}

Context:
{context[:1000]}

Answer with only "YES" or "NO"."""
            
            try:
                llm_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": verification_prompt}],
                    max_tokens=10,
                    temperature=0
                )
                answer = llm_response.choices[0].message.content.strip().upper()
                if answer == "NO":
                    verified_unsupported.append(claim)
            except Exception:
                # If LLM verification fails, keep original result
                verified_unsupported.append(claim)
        
        unsupported_claims = verified_unsupported
        hallucination_score = len(unsupported_claims) / total_claims if total_claims > 0 else 0.0
    
    return {
        'has_hallucinations': len(unsupported_claims) > 0,
        'hallucination_score': hallucination_score,
        'unsupported_claims': unsupported_claims,
        'supported_claims': supported_claims,
        'verification_details': verification_details,
        'total_claims': total_claims,
        'supported_count': len(supported_claims),
        'unsupported_count': len(unsupported_claims)
    }

