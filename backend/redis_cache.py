"""
Redis caching utility for improving search performance and reducing costs.

This module provides:
- Query embedding caching (reduces OpenAI API calls)
- Search result caching (reduces database queries)
- PDF document list caching
- Cache invalidation strategies
- TTL management
"""

import json
import hashlib
import pickle
from typing import Optional, List, Dict, Any, Tuple
from functools import wraps
import os
from dotenv import load_dotenv

load_dotenv()

# Try to import redis, but make it optional
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: redis package not installed. Caching will be disabled.")


class RedisCache:
    """Redis cache manager with connection pooling and error handling"""
    
    def __init__(self):
        self.redis_client = None
        self.enabled = False
        
        if not REDIS_AVAILABLE:
            return
        
        try:
            # Get Redis configuration from environment
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            redis_db = int(os.getenv("REDIS_DB", 0))
            redis_password = os.getenv("REDIS_PASSWORD", None)
            
            # Create Redis connection
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            print(f"Redis cache enabled: {redis_host}:{redis_port}/{redis_db}")
            
        except Exception as e:
            print(f"Warning: Redis connection failed. Caching disabled: {e}")
            self.enabled = False
            self.redis_client = None
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from prefix and arguments"""
        # Create a hash of the arguments
        key_data = json.dumps({
            'args': args,
            'kwargs': sorted(kwargs.items())
        }, sort_keys=True)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage in Redis"""
        try:
            # Try JSON first (for simple types)
            return json.dumps(value).encode('utf-8')
        except (TypeError, ValueError):
            # Fall back to pickle for complex types
            return pickle.dumps(value)
    
    def _deserialize(self, value: bytes) -> Any:
        """Deserialize value from Redis"""
        try:
            # Try JSON first
            return json.loads(value.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(value)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value is None:
                return None
            return self._deserialize(value)
        except Exception as e:
            print(f"Redis get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL (default: 1 hour)"""
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            serialized = self._serialize(value)
            return self.redis_client.setex(key, ttl, serialized)
        except Exception as e:
            print(f"Redis set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            print(f"Redis delete error for key {key}: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        if not self.enabled or not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            print(f"Redis delete_pattern error for pattern {pattern}: {e}")
            return 0
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            print(f"Redis exists error for key {key}: {e}")
            return False
    
    def get_or_set(self, key: str, func, ttl: int = 3600, *args, **kwargs) -> Any:
        """Get value from cache, or compute and cache it if not found"""
        if not self.enabled or not self.redis_client:
            # If Redis is not available, just call the function
            return func(*args, **kwargs)
        
        # Try to get from cache
        cached = self.get(key)
        if cached is not None:
            return cached
        
        # Compute value
        value = func(*args, **kwargs)
        
        # Cache it
        if value is not None:
            self.set(key, value, ttl)
        
        return value


# Global Redis cache instance
_redis_cache = None


def get_redis_cache() -> RedisCache:
    """Get or create global Redis cache instance"""
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCache()
    return _redis_cache


# Cache key prefixes
CACHE_PREFIX_EMBEDDING = "embedding"
CACHE_PREFIX_SEARCH = "search"
CACHE_PREFIX_PDF_LIST = "pdf_list"
CACHE_PREFIX_QUERY_EXPANSION = "query_expansion"


def cache_query_embedding(query: str, embedding: List[float], ttl: int = 86400) -> bool:
    """
    Cache a query embedding.
    
    Args:
        query: Query string
        embedding: Embedding vector
        ttl: Time to live in seconds (default: 24 hours)
    
    Returns:
        True if cached successfully
    """
    cache = get_redis_cache()
    key = f"{CACHE_PREFIX_EMBEDDING}:{hashlib.md5(query.encode()).hexdigest()}"
    return cache.set(key, embedding, ttl)


def get_cached_query_embedding(query: str) -> Optional[List[float]]:
    """
    Get cached query embedding.
    
    Args:
        query: Query string
    
    Returns:
        Cached embedding or None
    """
    cache = get_redis_cache()
    key = f"{CACHE_PREFIX_EMBEDDING}:{hashlib.md5(query.encode()).hexdigest()}"
    return cache.get(key)


def cache_search_results(
    query: str,
    filename: Optional[str],
    top_k: int,
    results: List[Tuple[str, str, int, float]],
    ttl: int = 3600
) -> bool:
    """
    Cache search results.
    
    Args:
        query: Query string
        filename: Optional filename filter
        top_k: Number of results
        results: Search results list
        ttl: Time to live in seconds (default: 1 hour)
    
    Returns:
        True if cached successfully
    """
    cache = get_redis_cache()
    # Include all parameters in key generation
    key_data = f"{query}:{filename or 'all'}:{top_k}"
    key = f"{CACHE_PREFIX_SEARCH}:{hashlib.md5(key_data.encode()).hexdigest()}"
    return cache.set(key, results, ttl)


def get_cached_search_results(
    query: str,
    filename: Optional[str],
    top_k: int
) -> Optional[List[Tuple[str, str, int, float]]]:
    """
    Get cached search results.
    
    Args:
        query: Query string
        filename: Optional filename filter
        top_k: Number of results
    
    Returns:
        Cached results or None
    """
    cache = get_redis_cache()
    key_data = f"{query}:{filename or 'all'}:{top_k}"
    key = f"{CACHE_PREFIX_SEARCH}:{hashlib.md5(key_data.encode()).hexdigest()}"
    return cache.get(key)


def cache_pdf_list(pdf_list: List[Dict[str, Any]], ttl: int = 300) -> bool:
    """
    Cache PDF document list.
    
    Args:
        pdf_list: List of PDF documents
        ttl: Time to live in seconds (default: 5 minutes)
    
    Returns:
        True if cached successfully
    """
    cache = get_redis_cache()
    key = CACHE_PREFIX_PDF_LIST
    return cache.set(key, pdf_list, ttl)


def get_cached_pdf_list() -> Optional[List[Dict[str, Any]]]:
    """
    Get cached PDF document list.
    
    Returns:
        Cached PDF list or None
    """
    cache = get_redis_cache()
    key = CACHE_PREFIX_PDF_LIST
    return cache.get(key)


def invalidate_pdf_cache():
    """Invalidate PDF-related caches"""
    cache = get_redis_cache()
    cache.delete(CACHE_PREFIX_PDF_LIST)
    # Also invalidate all search caches (they might reference old PDFs)
    cache.delete_pattern(f"{CACHE_PREFIX_SEARCH}:*")


def invalidate_search_cache(filename: Optional[str] = None):
    """
    Invalidate search caches.
    
    Args:
        filename: If provided, only invalidate caches for this filename
    """
    cache = get_redis_cache()
    if filename:
        # Invalidate all search caches (we can't easily filter by filename in the key)
        cache.delete_pattern(f"{CACHE_PREFIX_SEARCH}:*")
    else:
        cache.delete_pattern(f"{CACHE_PREFIX_SEARCH}:*")


def cache_query_expansion(query: str, variations: List[str], ttl: int = 3600) -> bool:
    """
    Cache query expansion results.
    
    Args:
        query: Original query
        variations: Query variations
        ttl: Time to live in seconds (default: 1 hour)
    
    Returns:
        True if cached successfully
    """
    cache = get_redis_cache()
    key = f"{CACHE_PREFIX_QUERY_EXPANSION}:{hashlib.md5(query.encode()).hexdigest()}"
    return cache.set(key, variations, ttl)


def get_cached_query_expansion(query: str) -> Optional[List[str]]:
    """
    Get cached query expansion results.
    
    Args:
        query: Original query
    
    Returns:
        Cached variations or None
    """
    cache = get_redis_cache()
    key = f"{CACHE_PREFIX_QUERY_EXPANSION}:{hashlib.md5(query.encode()).hexdigest()}"
    return cache.get(key)


def clear_all_cache() -> Dict[str, Any]:
    """
    Clear all Redis cache entries.
    
    Returns:
        Dictionary with operation result
    """
    cache = get_redis_cache()
    if not cache.enabled:
        return {
            "success": False,
            "message": "Redis cache is not enabled"
        }
    
    try:
        # Get all keys
        all_keys = cache.redis_client.keys("*")
        deleted_count = 0
        
        if all_keys:
            deleted_count = cache.redis_client.delete(*all_keys)
        
        return {
            "success": True,
            "message": f"Cleared {deleted_count} cache entries",
            "keys_deleted": deleted_count
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def clear_search_cache() -> Dict[str, Any]:
    """
    Clear all search-related cache entries.
    
    Returns:
        Dictionary with operation result
    """
    cache = get_redis_cache()
    if not cache.enabled:
        return {
            "success": False,
            "message": "Redis cache is not enabled"
        }
    
    try:
        deleted_count = cache.delete_pattern(f"{CACHE_PREFIX_SEARCH}:*")
        return {
            "success": True,
            "message": f"Cleared search cache",
            "keys_deleted": deleted_count
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Dictionary with cache statistics
    """
    cache = get_redis_cache()
    if not cache.enabled:
        return {
            "enabled": False,
            "message": "Redis cache is not enabled"
        }
    
    try:
        info = cache.redis_client.info()
        return {
            "enabled": True,
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "0B"),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "total_keys": len(cache.redis_client.keys("*"))
        }
    except Exception as e:
        return {
            "enabled": True,
            "error": str(e)
        }
