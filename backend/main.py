from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, Response
import json
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import time
import uuid
from openai import OpenAI
from typing import Optional, List, Dict, Any
from fastapi import Query
from database import search_similar_chunks, get_db_connection, get_all_pdf_documents, insert_or_update_pdf_document
from pdf_processor import process_pdf_to_chunks
from ingest_pdf import get_embeddings
import tempfile
import shutil
from reranking import rerank_chunks
from hybrid_search import hybrid_search
from query_expansion import expand_query
from context_assembly import assemble_context, optimize_context_order, deduplicate_chunks
from utils.logger import get_logger, calculate_token_cost
from utils.error_handling import retry_with_backoff, CircuitBreaker
from utils.hallucination_detection import detect_hallucinations
from prompts.templates import build_rag_prompt, PromptStrategy
from prompts.strategies import select_strategy, get_strategy_from_preset
from config import RAGConfig
from metrics import get_metrics_collector
import statistics
from redis_cache import (
    get_cached_query_embedding, cache_query_embedding,
    get_cached_query_expansion, cache_query_expansion,
    get_cached_pdf_list, cache_pdf_list, invalidate_pdf_cache,
    get_cached_search_results, cache_search_results,
    get_cache_stats, clear_all_cache, clear_search_cache,
    get_redis_cache
)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Learning Backend API")

# Configure CORS to allow Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure rate limiting (after CORS, before routes)
# Get Redis cache for rate limiting (optional, falls back to in-memory)
from rate_limiting import RateLimitMiddleware, RateLimiter
redis_cache = get_redis_cache()
rate_limiter = RateLimiter(redis_cache=redis_cache)
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

# Initialize OpenAI client with fail-fast validation
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "CRITICAL: OPENAI_API_KEY not found in environment variables. "
        "Please set OPENAI_API_KEY in your .env file before starting the server."
    )

try:
    client = OpenAI(api_key=api_key)
    # Validate the API key by making a minimal API call
    # This ensures the key is valid before the server starts
    # Just get the first model from the list to validate the key
    list(client.models.list())
except Exception as e:
    raise RuntimeError(
        f"CRITICAL: Invalid OpenAI API key or connection failed. "
        f"Please check your OPENAI_API_KEY. Error: {e}"
    )

# Initialize structured logger
logger = get_logger("rag_system")

# Initialize circuit breakers
openai_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=Exception
)

database_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=30,
    expected_exception=Exception
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-3.5-turbo"
    use_rag: bool = True  # Enable RAG by default
    filename: Optional[str] = None  # Optional: filter by PDF filename
    use_functions: bool = False  # Enable function calling
    use_reranking: bool = True  # Enable reranking for better retrieval quality
    rerank_strategy: str = "combined"  # Reranking strategy: "threshold", "keyword", "diversity", "length", "combined"
    prompt_strategy: Optional[str] = None  # Prompt strategy: "strict", "conversational", "technical", "summarize", "qna", or preset like "default", "accurate"
    use_hybrid_search: bool = False  # Enable hybrid search (vector + keyword)
    vector_weight: float = 0.7  # Weight for vector similarity in hybrid search (0.0-1.0)
    keyword_weight: float = 0.3  # Weight for keyword matching in hybrid search (0.0-1.0)
    use_query_expansion: bool = False  # Enable query expansion (generate variations)
    detect_hallucinations: bool = False  # Enable hallucination detection
    use_llm_verification: bool = False  # Use LLM for hallucination verification (more accurate)

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = []  # RAG sources with citations (default to empty list)
    function_calls: Optional[List[Dict[str, Any]]] = None  # Function calls made
    hallucination_detection: Optional[Dict[str, Any]] = None  # Hallucination detection results
    query_variations: Optional[List[str]] = None  # Query variations used (if expansion enabled)

@app.get("/")
async def root():
    return {"message": "AI Learning Backend API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/cache/stats")
async def get_cache_stats_endpoint():
    """
    Get Redis cache statistics
    """
    return get_cache_stats()

@app.post("/api/cache/clear")
async def clear_cache_endpoint(clear_all: bool = False):
    """
    Clear Redis cache
    
    Args:
        clear_all: If True, clear all cache. If False, only clear search cache.
    
    Returns:
        Operation result
    """
    from redis_cache import clear_all_cache, clear_search_cache
    
    if clear_all:
        result = clear_all_cache()
    else:
        result = clear_search_cache()
    
    logger.info(
        "cache_cleared",
        clear_all=clear_all,
        result=result
    )
    
    return result

@app.get("/api/config")
async def get_config_endpoint():
    """
    Get current RAG configuration values
    """
    return {
        "config": RAGConfig.get_config_summary(),
        "note": "These values can be overridden via environment variables. See env.example for details."
    }

@app.get("/api/metrics/retrieval")
async def get_retrieval_metrics(
    hours: int = Query(24, ge=1, le=168),
    use_reranking: Optional[bool] = Query(None),
    rerank_strategy: Optional[str] = Query(None)
):
    """
    Get aggregated retrieval quality metrics.
    
    Args:
        hours: Number of hours to look back (default: 24, max: 168)
        use_reranking: Filter by reranking usage (optional)
        rerank_strategy: Filter by reranking strategy (optional)
    
    Returns:
        Aggregated metrics including similarity scores, retrieval times, and quality distribution
    """
    metrics_collector = get_metrics_collector()
    metrics = metrics_collector.get_aggregated_metrics(
        hours=hours,
        use_reranking=use_reranking,
        rerank_strategy=rerank_strategy
    )
    
    return metrics

@app.get("/api/debug/retrieval")
async def debug_retrieval(
    query: str = Query("what are mammals?"),
    filename: Optional[str] = Query("animal_kingdom.pdf")
):
    """
    Debug endpoint to test retrieval and see what's happening with sources.
    """
    request_id = "debug_test"
    
    if not client:
        return {"error": "OpenAI client not initialized. Check OPENAI_API_KEY.", "client_is_none": True}
    
    try:
        # Step 1: Get embedding
        print(f"[DEBUG] Getting embedding for: {query}")
        query_embedding = get_query_embedding(query, request_id)
        if not query_embedding:
            return {
                "error": "Failed to get embedding", 
                "client_available": client is not None,
                "api_key_set": bool(os.getenv("OPENAI_API_KEY"))
            }
        print(f"[DEBUG] Embedding generated: {len(query_embedding)} dims")
        
        # Step 2: Search
        vector_results = search_similar_chunks(query_embedding, filename=filename, limit=10)
        
        # Step 3: Rerank
        from reranking import rerank_chunks
        reranked = rerank_chunks(
            query=query,
            results=vector_results,
            strategy="combined",
            top_k=3,
            min_similarity=RAGConfig.MIN_SIMILARITY_THRESHOLD
        )
        
        # Step 4: Assemble context
        context, sources, tokens = assemble_context(
            results=reranked,
            max_tokens=RAGConfig.CONTEXT_MAX_TOKENS,
            model="gpt-3.5-turbo",
            deduplicate=True,
            prioritize_high_similarity=True,
            reserve_tokens=RAGConfig.CONTEXT_RESERVE_TOKENS
        )
        
        # Step 5: Full build_rag_context
        full_context, full_sources, variations = build_rag_context(
            query=query,
            filename=filename,
            top_k=3,
            use_reranking=True,
            rerank_strategy="combined",
            request_id=request_id,
            model="gpt-3.5-turbo"
        )
        
        return {
            "query": query,
            "filename": filename,
            "vector_results_count": len(vector_results),
            "reranked_count": len(reranked),
            "context_length": len(context),
            "sources_count": len(sources),
            "full_context_length": len(full_context),
            "full_sources_count": len(full_sources),
            "vector_results_sample": str(vector_results[0])[:200] if vector_results else None,
            "reranked_sample": str(reranked[0])[:200] if reranked else None,
            "sources": sources,
            "full_sources": full_sources,
            "context_preview": context[:500] if context else None,
            "full_context_preview": full_context[:500] if full_context else None
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@retry_with_backoff(max_retries=3, initial_delay=1.0, exceptions=(Exception,))
def get_query_embedding(query: str, request_id: str = "unknown") -> list:
    """Generate embedding for a query string with retry logic and Redis caching"""
    if not client:
        return None
    
    # Try to get from cache first
    cached_embedding = get_cached_query_embedding(query)
    if cached_embedding is not None:
        logger.info(
            "embedding_cache_hit",
            request_id=request_id,
            query_preview=query[:100]
        )
        return cached_embedding
    
    start_time = time.time()
    try:
        # Use circuit breaker for OpenAI calls
        def _get_embedding():
            return client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
        
        response = openai_circuit_breaker.call(_get_embedding)
        embedding = response.data[0].embedding
        duration_ms = (time.time() - start_time) * 1000
        
        # Cache the embedding (24 hour TTL)
        cache_query_embedding(query, embedding, ttl=86400)
        
        # Log embedding generation
        logger.log_embedding_generation(
            request_id=request_id,
            query=query,
            model="text-embedding-ada-002",
            duration_ms=duration_ms,
            embedding_dim=len(embedding)
        )
        
        return embedding
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            "embedding_generation_error",
            request_id=request_id,
            query_preview=query[:100],
            duration_ms=duration_ms,
            error=e
        )
        return None

def build_rag_context(query: str, filename: Optional[str] = None, top_k: int = 3, 
                     use_reranking: bool = True, rerank_strategy: str = "combined",
                     use_hybrid_search: bool = False, vector_weight: float = 0.7,
                     keyword_weight: float = 0.3, use_query_expansion: bool = False,
                     request_id: str = "unknown", model: str = "gpt-3.5-turbo") -> tuple:
    """
    Retrieve relevant context using RAG with optional reranking, hybrid search, and query expansion
    
    Args:
        query: User query string
        filename: Optional filename filter
        top_k: Number of chunks to retrieve
        use_reranking: Enable reranking (default: True)
        rerank_strategy: Reranking strategy ("threshold", "keyword", "diversity", "length", "combined")
        use_hybrid_search: Enable hybrid search combining vector + keyword (default: False)
        vector_weight: Weight for vector similarity in hybrid search (default: 0.7)
        keyword_weight: Weight for keyword matching in hybrid search (default: 0.3)
        use_query_expansion: Enable query expansion to generate variations (default: False)
        request_id: Request ID for logging
    
    Returns:
        tuple: (context_string, sources_list, query_variations)
    """
    start_time = time.time()
    
    # Edge case: Empty or very short query
    if not query or len(query.strip()) < 2:
        logger.warning(
            "empty_query",
            request_id=request_id,
            query=query
        )
        return "", [], [query]
    
    # Normalize query
    query = query.strip()
    query_variations = [query]  # Track query variations
    
    # Edge case: Validate top_k
    top_k = max(1, min(top_k, RAGConfig.MAX_TOP_K))  # Clamp between 1 and MAX_TOP_K
    
    # Query expansion if enabled
    if use_query_expansion and client:
        try:
            # Try to get from cache first
            cached_variations = get_cached_query_expansion(query)
            if cached_variations is not None:
                query_variations = cached_variations
                logger.info(
                    "query_expansion_cache_hit",
                    request_id=request_id,
                    original_query=query,
                    variations=query_variations,
                    num_variations=len(query_variations)
                )
            else:
                variations = expand_query(query, num_variations=2, client=client)
                query_variations = variations
                # Cache the variations (1 hour TTL)
                cache_query_expansion(query, variations, ttl=3600)
                logger.info(
                    "query_expansion",
                    request_id=request_id,
                    original_query=query,
                    variations=variations,
                    num_variations=len(variations)
                )
        except Exception as e:
            logger.warning(
                "query_expansion_failed",
                request_id=request_id,
                query=query,
                error=str(e)
            )
            # Continue with original query if expansion fails
    
    # Check cache for final search results (cache key includes all parameters)
    # Note: We cache after all processing (hybrid search + reranking) for maximum benefit
    cache_key_params = {
        'query': query,
        'filename': filename,
        'top_k': top_k,
        'use_hybrid_search': use_hybrid_search,
        'vector_weight': vector_weight,
        'keyword_weight': keyword_weight,
        'use_reranking': use_reranking,
        'rerank_strategy': rerank_strategy
    }
    
    # Try to get cached final results
    cached_results = get_cached_search_results(
        query=query,
        filename=filename,
        top_k=top_k
    )
    
    # Note: We can't easily cache the full results with all parameters in the key
    # So we'll cache at the vector search level instead (before hybrid/reranking)
    # This still provides significant cost savings
    
    # Generate query embedding (use first query variation)
    primary_query = query_variations[0]
    query_embedding = get_query_embedding(primary_query, request_id)
    if not query_embedding:
        return "", [], query_variations
    
    # Search for similar chunks (retrieve more initially if reranking or hybrid search)
    initial_limit = top_k * RAGConfig.RERANK_TOP_K_MULTIPLIER if (use_reranking or use_hybrid_search or use_query_expansion) else top_k
    
    try:
        # Get vector search results with error handling
        def _search_chunks():
            return search_similar_chunks(query_embedding, filename=filename, limit=initial_limit)
        
        try:
            vector_results = database_circuit_breaker.call(_search_chunks)
        except Exception as e:
            logger.error(
                "vector_search_error",
                request_id=request_id,
                error=e
            )
            # Fallback: try without circuit breaker
            try:
                vector_results = search_similar_chunks(query_embedding, filename=filename, limit=initial_limit)
            except:
                return "", [], query_variations
        
        if not vector_results:
            return "", [], query_variations
        
        # Apply hybrid search if enabled
        if use_hybrid_search:
            results = hybrid_search(
                query=query,
                vector_results=vector_results,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                top_k=initial_limit
            )
        else:
            results = vector_results
        
        # Track chunks before reranking for metrics
        chunks_before_rerank = len(results) if use_reranking else None
        avg_similarity_before_rerank = None
        if use_reranking and results:
            similarities_before = [r[3] for r in results]
            avg_similarity_before_rerank = sum(similarities_before) / len(similarities_before) if similarities_before else None
        
        # Store original results before reranking for fallback
        original_results_before_rerank = results.copy() if results else []
        
        # Apply reranking if enabled
        if use_reranking:
            results_before_rerank = len(results)
            results = rerank_chunks(
                query=query,
                results=results,
                strategy=rerank_strategy,
                top_k=top_k,
                min_similarity=RAGConfig.MIN_SIMILARITY_THRESHOLD
            )
            results_after_rerank = len(results)
            
            # If all results filtered by threshold, use lower threshold as fallback
            if results_after_rerank == 0 and results_before_rerank > 0:
                logger.warning(
                    "all_results_filtered_by_reranking",
                    request_id=request_id,
                    query_preview=query[:100],
                    results_before=results_before_rerank,
                    min_similarity_threshold=RAGConfig.MIN_SIMILARITY_THRESHOLD,
                    top_similarities_before=[r[3] for r in original_results_before_rerank[:5]] if original_results_before_rerank else []
                )
                # Try with lower threshold (0.5 instead of 0.7) to ensure we get some results
                results = rerank_chunks(
                    query=query,
                    results=original_results_before_rerank,
                    strategy=rerank_strategy,
                    top_k=top_k,
                    min_similarity=0.5  # Lower threshold as fallback
                )
                logger.info(
                    "fallback_to_lower_threshold",
                    request_id=request_id,
                    fallback_results_count=len(results),
                    threshold_used=0.5
                )
        else:
            # Just take top_k if no reranking
            results = results[:top_k]
        
        # Edge case: No results found
        if not results:
            logger.warning(
                "no_results_found",
                request_id=request_id,
                query=query[:100],
                filename=filename,
                vector_results_count=len(vector_results) if 'vector_results' in locals() else 0,
                after_reranking=True
            )
            return "", [], query_variations
        
        # Log results before context assembly for debugging
        logger.info(
            "results_before_context_assembly",
            request_id=request_id,
            results_count=len(results),
            top_similarities=[r[3] for r in results[:3]] if results else [],
            query_preview=query[:100],
            sample_result=results[0] if results else None,
            results_type=str(type(results)),
            first_result_type=str(type(results[0])) if results else "no_results",
            first_result_preview=str(results[0])[:200] if results else None
        )
        
        # Store original results before optimization for fallback
        original_results_for_fallback = results.copy() if results else []
        
        # Optimize context order for better flow
        results = optimize_context_order(results, query)
        
        # Build context with smart assembly (deduplication + token management)
        # Ensure we pass all required parameters explicitly
        assembly_result = assemble_context(
            results=results,
            max_tokens=RAGConfig.CONTEXT_MAX_TOKENS,
            model=model,
            deduplicate=True,
            prioritize_high_similarity=True,
            reserve_tokens=RAGConfig.CONTEXT_RESERVE_TOKENS
        )
        
        # Unpack result and verify
        if isinstance(assembly_result, tuple) and len(assembly_result) >= 3:
            context, sources, context_tokens = assembly_result[0], assembly_result[1], assembly_result[2]
        else:
            logger.error(
                "invalid_assemble_context_return",
                request_id=request_id,
                return_type=str(type(assembly_result)),
                return_value=str(assembly_result)[:200]
            )
            context, sources, context_tokens = "", [], 0
        
        # CRITICAL: Verify sources immediately after assembly
        logger.info(
            "context_assembly_complete",
            request_id=request_id,
            query_preview=query[:100],
            results_count=len(results),
            context_length=len(context) if context else 0,
            sources_count=len(sources) if sources else 0,
            sources_type=type(sources).__name__ if sources else "None",
            sources_is_list=isinstance(sources, list),
            context_tokens=context_tokens,
            sources_sample=sources[0] if sources and len(sources) > 0 else None,
            assembly_return_type=str(type(assembly_result)),
            assembly_return_length=len(assembly_result) if isinstance(assembly_result, (list, tuple)) else "N/A"
        )
        
        # CRITICAL: If sources is not a list, fix it
        if not isinstance(sources, list):
            logger.error(
                "sources_not_list_after_assembly",
                request_id=request_id,
                sources_type=str(type(sources)),
                sources_value=str(sources)[:200]
            )
            sources = []
        
        # CRITICAL FIX: If context exists but sources are empty, rebuild sources from results
        if context and len(context) > 0 and (not sources or len(sources) == 0):
            logger.error(
                "sources_empty_but_context_exists",
                request_id=request_id,
                query_preview=query[:100],
                context_length=len(context),
                results_count=len(results),
                results_type=str(type(results)),
                first_result_type=str(type(results[0])) if results else "no_results",
                first_result_preview=str(results[0])[:200] if results else None,
                rebuilding_sources=True
            )
            # Rebuild sources from results - this should never happen but is a safety net
            sources = []
            # Use original results before optimization (stored above)
            rebuild_results = original_results_for_fallback if original_results_for_fallback else results
            for i, result in enumerate(rebuild_results[:top_k * 2], 1):  # Get more to ensure we have enough
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
                        logger.warning(
                            "invalid_result_format_in_rebuild",
                            request_id=request_id,
                            result_index=i-1,
                            result_type=str(type(result)),
                            result_length=len(result) if isinstance(result, (list, tuple)) else "N/A"
                        )
                except Exception as e:
                    logger.error(
                        "error_rebuilding_source",
                        request_id=request_id,
                        result_index=i-1,
                        error=str(e),
                        result_preview=str(result)[:200]
                    )
            logger.info(
                "sources_rebuilt_from_results",
                request_id=request_id,
                rebuilt_sources_count=len(sources),
                results_used=min(len(results), top_k * 2)
            )
        
        # CRITICAL: Ensure sources is never None - always return a list
        if sources is None:
            sources = []
            logger.error(
                "sources_was_none",
                request_id=request_id,
                query_preview=query[:100]
            )
        
        # Edge case: Empty context after assembly
        # CRITICAL: Only return early if context is empty. If context exists, keep sources even if empty (will rebuild)
        if not context:
            logger.warning(
                "empty_context_after_assembly",
                request_id=request_id,
                query=query[:100],
                results_count=len(results),
                context_length=0,
                sources_count=len(sources) if sources else 0
            )
            return "", [], query_variations
        
        # If context exists but sources are empty, we already rebuilt them above
        # So just log a warning but continue
        if not sources or len(sources) == 0:
            logger.warning(
                "sources_empty_but_context_exists_after_assembly",
                request_id=request_id,
                query=query[:100],
                context_length=len(context),
                results_count=len(results),
                note="Sources should have been rebuilt above, but are still empty"
            )
            # Don't return - continue with empty sources, they should have been rebuilt
        
        # Extract similarities for enhanced logging
        similarities = [s["similarity"] for s in sources]
        retrieval_time_ms = (time.time() - start_time) * 1000
        
        # Calculate enhanced metrics
        avg_similarity = sum(similarities) / len(similarities) if similarities else None
        min_similarity = min(similarities) if similarities else None
        max_similarity = max(similarities) if similarities else None
        similarity_std = statistics.stdev(similarities) if len(similarities) > 1 else None
        
        # Calculate reranking impact
        reranking_impact = None
        if use_reranking and avg_similarity_before_rerank is not None and avg_similarity is not None:
            reranking_impact = avg_similarity - avg_similarity_before_rerank
        
        # Calculate quality distribution
        high_quality_chunks = sum(1 for s in similarities if s >= 0.8) if similarities else None
        medium_quality_chunks = sum(1 for s in similarities if 0.7 <= s < 0.8) if similarities else None
        low_quality_chunks = sum(1 for s in similarities if s < 0.7) if similarities else None
        
        # Log retrieval with enhanced metrics
        logger.log_retrieval(
            request_id=request_id,
            query=query,
            top_k=top_k,
            chunks_retrieved=len(results),
            retrieval_time_ms=retrieval_time_ms,
            use_reranking=use_reranking,
            rerank_strategy=rerank_strategy if use_reranking else None,
            avg_similarity=avg_similarity,
            min_similarity=min_similarity,
            max_similarity=max_similarity,
            similarity_std=similarity_std,
            chunks_before_rerank=chunks_before_rerank,
            reranking_impact=reranking_impact,
            high_quality_chunks=high_quality_chunks,
            medium_quality_chunks=medium_quality_chunks,
            low_quality_chunks=low_quality_chunks
        )
        
        # Record metrics for observability
        metrics_collector = get_metrics_collector()
        metrics_collector.record_retrieval(
            request_id=request_id,
            query=query,
            top_k=top_k,
            chunks_retrieved=len(results),
            avg_similarity=avg_similarity,
            min_similarity=min_similarity,
            max_similarity=max_similarity,
            use_reranking=use_reranking,
            rerank_strategy=rerank_strategy if use_reranking else None,
            retrieval_time_ms=retrieval_time_ms,
            chunks_before_rerank=chunks_before_rerank,
            reranking_impact=reranking_impact
        )
        
        # Log hybrid search usage
        if use_hybrid_search:
            logger.info(
                "hybrid_search_used",
                request_id=request_id,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight
            )
        
        # FINAL SAFETY CHECK: Ensure sources match context
        # If we have context, we MUST have sources (one per context part)
        if context and len(context) > 0:
            # Count context parts by looking for [Context X] markers
            context_parts_count = context.count("[Context")
            if len(sources) != context_parts_count and context_parts_count > 0:
                logger.error(
                    "sources_count_mismatch_before_return",
                    request_id=request_id,
                    context_parts_count=context_parts_count,
                    sources_count=len(sources),
                    query_preview=query[:100],
                    rebuilding_sources=True
                )
                # Rebuild sources to match context parts
                if original_results_for_fallback and len(original_results_for_fallback) > 0:
                    sources = []
                    for i, result in enumerate(original_results_for_fallback[:context_parts_count], 1):
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
                        except Exception as e:
                            logger.error(f"Error in final source rebuild {i}: {e}")
                    logger.info(
                        "sources_rebuilt_final_check",
                        request_id=request_id,
                        final_sources_count=len(sources),
                        context_parts_count=context_parts_count
                    )
        
        # Ensure sources is a list
        if not isinstance(sources, list):
            logger.error(f"sources_not_list_at_return: {type(sources)}")
            sources = []
        
        return context, sources, query_variations
    except Exception as e:
        retrieval_time_ms = (time.time() - start_time) * 1000
        logger.error(
            "rag_retrieval_error",
            request_id=request_id,
            query_preview=query[:100],
            retrieval_time_ms=retrieval_time_ms,
            error=e
        )
        return "", [], [query]

def get_function_definitions() -> List[Dict[str, Any]]:
    """Define available functions for function calling"""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit for temperature"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "A mathematical expression to evaluate, e.g. '2 + 2' or 'sqrt(16)'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current date and time",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    ]

def execute_function_call(function_name: str, arguments: Dict[str, Any]) -> str:
    """Execute a function call and return the result"""
    import json
    from datetime import datetime
    import math
    
    try:
        if function_name == "get_weather":
            # Mock weather function - in production, call a real weather API
            location = arguments.get("location", "unknown")
            unit = arguments.get("unit", "fahrenheit")
            return f"The weather in {location} is sunny, 72°{unit[0].upper()}"
        
        elif function_name == "calculate":
            # Safe evaluation of mathematical expressions
            expression = arguments.get("expression", "")
            # Only allow safe math operations
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            try:
                result = eval(expression, {"__builtins__": {}}, allowed_names)
                return f"The result of {expression} is {result}"
            except Exception as e:
                return f"Error calculating {expression}: {str(e)}"
        
        elif function_name == "get_current_time":
            now = datetime.now()
            return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        
        else:
            return f"Unknown function: {function_name}"
    except Exception as e:
        return f"Error executing function {function_name}: {str(e)}"

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint to handle chat requests with RAG and Function Calling support
    
    Features:
    - RAG: Retrieves relevant context from PDF chunks when use_rag=True
    - Function Calling: Enables tool calling when use_functions=True
    - Prompt Templates: Uses configurable prompt strategies
    - Structured Logging: Logs all operations with timing and metrics
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Edge case: Validate request
        if not request.message or len(request.message.strip()) < 1:
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )
        
        # Edge case: Validate model name
        valid_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-4o", "gpt-4o-mini"]
        if request.model not in valid_models:
            logger.warning(
                "invalid_model",
                request_id=request_id,
                model=request.model,
                using_default="gpt-3.5-turbo"
            )
            request.model = "gpt-3.5-turbo"
        
        # Log incoming request
        logger.log_request(
            method="POST",
            path="/api/chat",
            request_id=request_id,
            body={
                "message_preview": request.message[:100],
                "model": request.model,
                "use_rag": request.use_rag,
                "use_functions": request.use_functions,
                "use_reranking": request.use_reranking,
                "prompt_strategy": request.prompt_strategy
            }
        )
        
        # Note: Client is validated at startup (fail-fast), so this should never happen
        # Keeping as a safety check in case of runtime issues
        if not client:
            raise HTTPException(
                status_code=500,
                detail="OpenAI client not initialized. This should not happen as the server validates the API key at startup."
            )
        
        messages = []
        sources = []
        function_calls_made = []
        
        # Determine prompt strategy
        if request.prompt_strategy:
            # Try preset first, then direct strategy name
            try:
                strategy = get_strategy_from_preset(request.prompt_strategy)
            except:
                try:
                    strategy = PromptStrategy(request.prompt_strategy)
                except:
                    strategy = PromptStrategy.CONVERSATIONAL
        else:
            # Auto-select based on query
            strategy = select_strategy(request.message)
        
        # Build prompts using template system
        query_variations = [request.message]
        context = ""
        sources = []
        if request.use_rag:
            context, sources, query_variations = build_rag_context(
                request.message, 
                filename=request.filename,
                use_reranking=request.use_reranking,
                rerank_strategy=request.rerank_strategy,
                use_hybrid_search=request.use_hybrid_search,
                vector_weight=request.vector_weight,
                keyword_weight=request.keyword_weight,
                use_query_expansion=request.use_query_expansion,
                request_id=request_id,
                model=request.model
            )
            # Log sources for debugging - CRITICAL DEBUG INFO
            logger.info(
                "rag_sources_retrieved",
                request_id=request_id,
                num_sources=len(sources) if sources else 0,
                sources_type=type(sources).__name__,
                has_context=bool(context),
                context_length=len(context) if context else 0,
                use_rag=True,
                sources_preview=sources[:2] if sources and len(sources) > 0 else None
            )
            
            # CRITICAL: Ensure sources is always a list, never None
            if sources is None:
                logger.error(
                    "sources_is_none_from_build_rag_context",
                    request_id=request_id,
                    query_preview=request.message[:100]
                )
                sources = []
            elif not isinstance(sources, list):
                logger.error(
                    "sources_is_not_list",
                    request_id=request_id,
                    sources_type=type(sources).__name__,
                    sources_value=str(sources)[:200]
                )
                sources = []
            
            system_content, user_content = build_rag_prompt(
                query=request.message,
                context=context,
                strategy=strategy,
                use_rag=True
            )
        else:
            logger.info(
                "rag_disabled",
                request_id=request_id,
                use_rag=False
            )
            
            system_content, user_content = build_rag_prompt(
                query=request.message,
                context="",
                strategy=strategy,
                use_rag=False
            )
        
        messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})
        
        # Prepare API call parameters
        api_params = {
            "model": request.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        # Add function calling if enabled
        if request.use_functions:
            api_params["tools"] = get_function_definitions()
            api_params["tool_choice"] = "auto"  # Let the model decide when to use functions
        
        # Call OpenAI API
        llm_start_time = time.time()
        response = client.chat.completions.create(**api_params)
        llm_duration_ms = (time.time() - llm_start_time) * 1000
        
        assistant_message = response.choices[0].message
        final_response = assistant_message.content or ""
        
        # Extract token usage
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        
        # Calculate cost
        cost_usd = calculate_token_cost(request.model, prompt_tokens, completion_tokens)
        
        # Log LLM call
        logger.log_llm_call(
            request_id=request_id,
            model=request.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_ms=llm_duration_ms,
            cost_usd=cost_usd
        )
        
        # Store token count for rate limiting middleware
        # This will be read by the middleware to track daily token usage
        token_count_for_tracking = total_tokens
        
        # Handle function calls if any
        if request.use_functions and assistant_message.tool_calls:
            # Add assistant's message with tool calls to conversation
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })
            
            # Execute function calls
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                try:
                    import json
                    arguments = json.loads(tool_call.function.arguments)
                except:
                    arguments = {}
                
                func_start_time = time.time()
                function_result = execute_function_call(function_name, arguments)
                func_duration_ms = (time.time() - func_start_time) * 1000
                
                function_calls_made.append({
                    "function": function_name,
                    "arguments": arguments,
                    "result": function_result
                })
                
                # Log function call
                logger.log_function_call(
                    request_id=request_id,
                    function_name=function_name,
                    arguments=arguments,
                    result=function_result,
                    duration_ms=func_duration_ms
                )
                
                # Add function result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": function_result
                })
            
            # Get final response after function execution
            final_llm_start = time.time()
            final_response_obj = client.chat.completions.create(
                model=request.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            final_llm_duration_ms = (time.time() - final_llm_start) * 1000
            final_response = final_response_obj.choices[0].message.content or ""
            
            # Log second LLM call if function was called
            final_usage = final_response_obj.usage
            if final_usage:
                final_cost = calculate_token_cost(
                    request.model, 
                    final_usage.prompt_tokens, 
                    final_usage.completion_tokens
                )
                logger.log_llm_call(
                    request_id=request_id,
                    model=request.model,
                    prompt_tokens=final_usage.prompt_tokens,
                    completion_tokens=final_usage.completion_tokens,
                    total_tokens=final_usage.total_tokens,
                    duration_ms=final_llm_duration_ms,
                    cost_usd=final_cost
                )
        
        # Hallucination detection if enabled
        hallucination_results = None
        if request.detect_hallucinations and request.use_rag and context:
            try:
                hallucination_results = detect_hallucinations(
                    response=final_response,
                    context=context,
                    sources=sources,
                    threshold=RAGConfig.HALLUCINATION_THRESHOLD,
                    use_llm_verification=request.use_llm_verification,
                    client=client
                )
                
                logger.info(
                    "hallucination_detection",
                    request_id=request_id,
                    has_hallucinations=hallucination_results.get('has_hallucinations', False),
                    hallucination_score=hallucination_results.get('hallucination_score', 0.0),
                    total_claims=hallucination_results.get('total_claims', 0),
                    unsupported_count=hallucination_results.get('unsupported_count', 0)
                )
            except Exception as e:
                logger.error(
                    "hallucination_detection_error",
                    request_id=request_id,
                    error=e
                )
                # Continue without hallucination detection if it fails
        
        # Calculate total response time
        total_time_ms = (time.time() - start_time) * 1000
        
        # Log response
        logger.log_response(
            request_id=request_id,
            status_code=200,
            response_time_ms=total_time_ms,
            response_size=len(final_response)
        )
        
        # Ensure sources is always a list - CRITICAL FIX
        if sources is None:
            sources_list = []
            logger.error(
                "sources_was_none_at_response_creation",
                request_id=request_id,
                use_rag=request.use_rag
            )
        elif not isinstance(sources, list):
            sources_list = []
            logger.error(
                "sources_was_not_list_at_response_creation",
                request_id=request_id,
                sources_type=type(sources).__name__,
                use_rag=request.use_rag
            )
        else:
            sources_list = sources
        
        # Log final response structure for debugging - CRITICAL DEBUG
        logger.info(
            "chat_response_ready",
            request_id=request_id,
            response_length=len(final_response),
            num_sources=len(sources_list),
            sources_type=type(sources_list).__name__,
            sources_is_list=isinstance(sources_list, list),
            num_function_calls=len(function_calls_made) if function_calls_made else 0,
            use_rag=request.use_rag,
            has_context=bool(context),
            context_length=len(context) if context else 0,
            sources_preview=sources_list[:1] if sources_list else None
        )
        
        # Create response with token tracking header
        response = ChatResponse(
            response=final_response,
            sources=sources_list,  # Always return a list, never None
            function_calls=function_calls_made if function_calls_made else None,
            hallucination_detection=hallucination_results,
            query_variations=query_variations if request.use_query_expansion else None
        )
        
        # FINAL CHECK: Log what we're actually returning
        logger.info(
            "response_created",
            request_id=request_id,
            response_sources_count=len(response.sources) if response.sources else 0,
            response_sources_type=type(response.sources).__name__ if response.sources else "None"
        )
        
        # Note: Token tracking is handled by middleware reading response
        # We can't directly set headers in Pydantic model, so middleware
        # will need to extract from response body or we track in request state
        # For now, token tracking happens in middleware via Redis/memory
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        total_time_ms = (time.time() - start_time) * 1000
        logger.error(
            "chat_endpoint_error",
            request_id=request_id,
            response_time_ms=total_time_ms,
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")

@app.get("/api/pdfs")
async def get_pdfs():
    """
    Get list of all available PDF documents with metadata from pdf_documents table
    Uses Redis cache to reduce database queries
    """
    try:
        # Try to get from cache first
        cached_pdfs = get_cached_pdf_list()
        if cached_pdfs is not None:
            return {
                "pdfs": [pdf['filename'] for pdf in cached_pdfs],
                "pdf_documents": cached_pdfs,
                "cached": True
            }
        
        # Fetch from database
        pdf_documents = get_all_pdf_documents()
        
        # Cache the results (5 minute TTL)
        cache_pdf_list(pdf_documents, ttl=300)
        
        # Return both simple list and detailed info
        return {
            "pdfs": [pdf['filename'] for pdf in pdf_documents],
            "pdf_documents": pdf_documents,
            "cached": False
        }
    except Exception as e:
        logger.error("get_pdfs_error", error=e)
        # Fallback to old method if pdf_documents table doesn't exist
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            query = "SELECT DISTINCT filename FROM pdf_chunks ORDER BY filename"
            cursor.execute(query)
            results = cursor.fetchall()
            
            pdfs = [row[0] for row in results]
            
            cursor.close()
            conn.close()
            
            return {"pdfs": pdfs, "pdf_documents": []}
        except Exception as fallback_error:
            logger.error("get_pdfs_fallback_error", error=fallback_error)
            raise HTTPException(status_code=500, detail=f"Error fetching PDFs: {str(fallback_error)}")

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF file
    
    Note: This endpoint may take several minutes for large PDFs due to:
    - PDF text extraction and chunking
    - Embedding generation (one API call per chunk)
    - Database insertion
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            logger.info(
                "pdf_upload_started",
                request_id=request_id,
                filename=file.filename
            )
            
            # Process PDF to chunks
            logger.info(f"Processing PDF: {file.filename}")
            chunks = process_pdf_to_chunks(
                tmp_path,
                chunk_method="tokens",
                chunk_size=500,
                overlap=50,
                use_contextual_chunking=False
            )
            
            if not chunks:
                raise HTTPException(status_code=400, detail="Failed to extract text from PDF")
            
            logger.info(f"Created {len(chunks)} chunks. Generating embeddings...")
            
            # Generate embeddings
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(status_code=500, detail="OpenAI API key not configured")
            
            # This can take a while - one API call per chunk
            chunks_with_embeddings = get_embeddings(chunks, api_key)
            
            logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks. Storing in database...")
            
            if not chunks_with_embeddings:
                raise HTTPException(status_code=500, detail="Failed to generate embeddings")
            
            # Store in database
            from database import insert_chunks
            insert_chunks(chunks_with_embeddings)
            
            # Update PDF document metadata (this is also done in insert_chunks, but we do it here for immediate update)
            insert_or_update_pdf_document(
                filename=file.filename,
                chunk_count=len(chunks_with_embeddings),
                display_name=file.filename,  # Can be customized later
                description=None,
                options=None
            )
            
            # Invalidate PDF cache since we added a new PDF
            invalidate_pdf_cache()
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Prepare response data
            response_data = {
                "message": "PDF uploaded and ingested successfully",
                "filename": file.filename,
                "chunks_created": len(chunks_with_embeddings),
                "processing_time_ms": round(processing_time_ms, 2)
            }
            
            logger.info(
                "pdf_upload_success",
                request_id=request_id,
                filename=file.filename,
                chunks_created=len(chunks_with_embeddings),
                processing_time_ms=processing_time_ms,
                response_data=response_data
            )
            
            # Return JSONResponse explicitly to ensure proper HTTP response
            # This ensures Angular HttpClient receives the response correctly
            return JSONResponse(content=response_data, status_code=200)
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        logger.error(
            "pdf_upload_error",
            request_id=request_id,
            filename=file.filename if file else "unknown",
            processing_time_ms=processing_time_ms,
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

