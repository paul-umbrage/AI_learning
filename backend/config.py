"""
Configuration module for RAG system thresholds and parameters.

Allows configuration via environment variables with sensible defaults.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class RAGConfig:
    """Configuration for RAG retrieval and reranking parameters."""
    
    # Reranking thresholds
    MIN_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_MIN_SIMILARITY_THRESHOLD", "0.7"))
    RERANK_TOP_K_MULTIPLIER: int = int(os.getenv("RAG_RERANK_TOP_K_MULTIPLIER", "3"))
    
    # Context assembly thresholds
    CONTEXT_MAX_TOKENS: int = int(os.getenv("RAG_CONTEXT_MAX_TOKENS", "2000"))
    CONTEXT_RESERVE_TOKENS: int = int(os.getenv("RAG_CONTEXT_RESERVE_TOKENS", "200"))
    DEDUPLICATION_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_DEDUP_SIMILARITY_THRESHOLD", "0.85"))
    
    # Retrieval parameters
    DEFAULT_TOP_K: int = int(os.getenv("RAG_DEFAULT_TOP_K", "3"))
    MAX_TOP_K: int = int(os.getenv("RAG_MAX_TOP_K", "20"))
    
    # Reranking strategy weights (for combined strategy)
    KEYWORD_WEIGHT: float = float(os.getenv("RAG_KEYWORD_WEIGHT", "0.35"))
    SIMILARITY_WEIGHT: float = float(os.getenv("RAG_SIMILARITY_WEIGHT", "0.65"))
    
    # Length filtering
    MIN_CHUNK_LENGTH: int = int(os.getenv("RAG_MIN_CHUNK_LENGTH", "50"))
    MAX_CHUNK_LENGTH: int = int(os.getenv("RAG_MAX_CHUNK_LENGTH", "2000"))
    OPTIMAL_CHUNK_LENGTH_MIN: int = int(os.getenv("RAG_OPTIMAL_CHUNK_MIN", "100"))
    OPTIMAL_CHUNK_LENGTH_MAX: int = int(os.getenv("RAG_OPTIMAL_CHUNK_MAX", "500"))
    
    # Diversity reranking
    MAX_CHUNKS_PER_PAGE: int = int(os.getenv("RAG_MAX_CHUNKS_PER_PAGE", "2"))
    MAX_CHUNKS_PER_DOCUMENT: Optional[int] = (
        int(os.getenv("RAG_MAX_CHUNKS_PER_DOCUMENT")) 
        if os.getenv("RAG_MAX_CHUNKS_PER_DOCUMENT") 
        else None
    )
    
    # Hallucination detection
    HALLUCINATION_THRESHOLD: float = float(os.getenv("RAG_HALLUCINATION_THRESHOLD", "0.7"))
    
    # Query expansion
    QUERY_EXPANSION_NUM_VARIATIONS: int = int(os.getenv("RAG_QUERY_EXPANSION_VARIATIONS", "2"))
    
    # Hybrid search defaults
    DEFAULT_VECTOR_WEIGHT: float = float(os.getenv("RAG_DEFAULT_VECTOR_WEIGHT", "0.7"))
    DEFAULT_KEYWORD_WEIGHT: float = float(os.getenv("RAG_DEFAULT_KEYWORD_WEIGHT", "0.3"))
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """Get a summary of all configuration values."""
        return {
            "reranking": {
                "min_similarity_threshold": cls.MIN_SIMILARITY_THRESHOLD,
                "top_k_multiplier": cls.RERANK_TOP_K_MULTIPLIER,
                "keyword_weight": cls.KEYWORD_WEIGHT,
                "similarity_weight": cls.SIMILARITY_WEIGHT,
            },
            "context_assembly": {
                "max_tokens": cls.CONTEXT_MAX_TOKENS,
                "reserve_tokens": cls.CONTEXT_RESERVE_TOKENS,
                "deduplication_threshold": cls.DEDUPLICATION_SIMILARITY_THRESHOLD,
            },
            "retrieval": {
                "default_top_k": cls.DEFAULT_TOP_K,
                "max_top_k": cls.MAX_TOP_K,
            },
            "length_filtering": {
                "min_chunk_length": cls.MIN_CHUNK_LENGTH,
                "max_chunk_length": cls.MAX_CHUNK_LENGTH,
                "optimal_range": [cls.OPTIMAL_CHUNK_LENGTH_MIN, cls.OPTIMAL_CHUNK_LENGTH_MAX],
            },
            "diversity": {
                "max_chunks_per_page": cls.MAX_CHUNKS_PER_PAGE,
                "max_chunks_per_document": cls.MAX_CHUNKS_PER_DOCUMENT,
            },
            "hallucination_detection": {
                "threshold": cls.HALLUCINATION_THRESHOLD,
            },
            "query_expansion": {
                "num_variations": cls.QUERY_EXPANSION_NUM_VARIATIONS,
            },
            "hybrid_search": {
                "default_vector_weight": cls.DEFAULT_VECTOR_WEIGHT,
                "default_keyword_weight": cls.DEFAULT_KEYWORD_WEIGHT,
            }
        }
