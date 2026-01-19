"""
Metrics collection and tracking for RAG system observability.

Provides endpoints and utilities for tracking retrieval quality metrics
over time.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json
from redis_cache import get_redis_cache


class RetrievalMetricsCollector:
    """Collects and aggregates retrieval quality metrics."""
    
    def __init__(self, redis_cache=None):
        """
        Initialize metrics collector.
        
        Args:
            redis_cache: Optional Redis cache instance for persistence
        """
        self.redis_cache = redis_cache or get_redis_cache()
        self.metrics_key_prefix = "rag:metrics:"
    
    def record_retrieval(
        self,
        request_id: str,
        query: str,
        top_k: int,
        chunks_retrieved: int,
        avg_similarity: Optional[float] = None,
        min_similarity: Optional[float] = None,
        max_similarity: Optional[float] = None,
        use_reranking: bool = False,
        rerank_strategy: Optional[str] = None,
        retrieval_time_ms: Optional[float] = None,
        chunks_before_rerank: Optional[int] = None,
        reranking_impact: Optional[float] = None
    ):
        """
        Record a retrieval operation for metrics tracking.
        
        Args:
            request_id: Request identifier
            query: User query
            top_k: Requested number of chunks
            chunks_retrieved: Actual chunks retrieved
            avg_similarity: Average similarity score
            min_similarity: Minimum similarity score
            max_similarity: Maximum similarity score
            use_reranking: Whether reranking was used
            rerank_strategy: Reranking strategy name
            retrieval_time_ms: Retrieval time in milliseconds
            chunks_before_rerank: Chunks before reranking
            reranking_impact: Improvement from reranking
        """
        timestamp = datetime.utcnow().isoformat()
        
        metric = {
            "request_id": request_id,
            "timestamp": timestamp,
            "query_preview": query[:100] if len(query) > 100 else query,
            "top_k": top_k,
            "chunks_retrieved": chunks_retrieved,
            "avg_similarity": avg_similarity,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity,
            "use_reranking": use_reranking,
            "rerank_strategy": rerank_strategy,
            "retrieval_time_ms": retrieval_time_ms,
            "chunks_before_rerank": chunks_before_rerank,
            "reranking_impact": reranking_impact
        }
        
        # Store in Redis if available (with 7 day TTL)
        if self.redis_cache:
            try:
                key = f"{self.metrics_key_prefix}{request_id}"
                self.redis_cache.setex(
                    key,
                    7 * 24 * 60 * 60,  # 7 days
                    json.dumps(metric)
                )
                
                # Also add to time-series list
                list_key = f"{self.metrics_key_prefix}list"
                self.redis_cache.lpush(list_key, json.dumps(metric))
                self.redis_cache.ltrim(list_key, 0, 9999)  # Keep last 10k metrics
                self.redis_cache.expire(list_key, 7 * 24 * 60 * 60)
            except Exception:
                # Silently fail if Redis is unavailable
                pass
    
    def get_aggregated_metrics(
        self,
        hours: int = 24,
        use_reranking: Optional[bool] = None,
        rerank_strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated metrics for the specified time period.
        
        Args:
            hours: Number of hours to look back (default: 24)
            use_reranking: Filter by reranking usage (optional)
            rerank_strategy: Filter by reranking strategy (optional)
        
        Returns:
            Dictionary with aggregated metrics
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        metrics = []
        if self.redis_cache:
            try:
                list_key = f"{self.metrics_key_prefix}list"
                raw_metrics = self.redis_cache.lrange(list_key, 0, 9999)
                
                for raw_metric in raw_metrics:
                    try:
                        metric = json.loads(raw_metric)
                        metric_time = datetime.fromisoformat(metric["timestamp"])
                        if metric_time >= cutoff_time:
                            # Apply filters
                            if use_reranking is not None and metric.get("use_reranking") != use_reranking:
                                continue
                            if rerank_strategy is not None and metric.get("rerank_strategy") != rerank_strategy:
                                continue
                            metrics.append(metric)
                    except Exception:
                        continue
            except Exception:
                pass
        
        if not metrics:
            return {
                "period_hours": hours,
                "total_requests": 0,
                "message": "No metrics available for this period"
            }
        
        # Calculate aggregations
        similarities = [m["avg_similarity"] for m in metrics if m.get("avg_similarity") is not None]
        retrieval_times = [m["retrieval_time_ms"] for m in metrics if m.get("retrieval_time_ms") is not None]
        chunks_retrieved = [m["chunks_retrieved"] for m in metrics]
        
        reranking_impacts = [m["reranking_impact"] for m in metrics if m.get("reranking_impact") is not None]
        
        # Count by reranking usage
        with_reranking = sum(1 for m in metrics if m.get("use_reranking"))
        without_reranking = len(metrics) - with_reranking
        
        # Count by strategy
        strategy_counts = defaultdict(int)
        for m in metrics:
            strategy = m.get("rerank_strategy") or "none"
            strategy_counts[strategy] += 1
        
        return {
            "period_hours": hours,
            "total_requests": len(metrics),
            "similarity_metrics": {
                "avg": sum(similarities) / len(similarities) if similarities else None,
                "min": min(similarities) if similarities else None,
                "max": max(similarities) if similarities else None,
                "count": len(similarities)
            },
            "retrieval_time_metrics": {
                "avg_ms": sum(retrieval_times) / len(retrieval_times) if retrieval_times else None,
                "min_ms": min(retrieval_times) if retrieval_times else None,
                "max_ms": max(retrieval_times) if retrieval_times else None,
                "count": len(retrieval_times)
            },
            "chunks_retrieved_metrics": {
                "avg": sum(chunks_retrieved) / len(chunks_retrieved) if chunks_retrieved else None,
                "min": min(chunks_retrieved) if chunks_retrieved else None,
                "max": max(chunks_retrieved) if chunks_retrieved else None
            },
            "reranking_metrics": {
                "with_reranking": with_reranking,
                "without_reranking": without_reranking,
                "avg_impact": sum(reranking_impacts) / len(reranking_impacts) if reranking_impacts else None,
                "strategy_distribution": dict(strategy_counts)
            },
            "quality_distribution": {
                "high_quality": sum(1 for s in similarities if s and s >= 0.8) if similarities else 0,
                "medium_quality": sum(1 for s in similarities if s and 0.7 <= s < 0.8) if similarities else 0,
                "low_quality": sum(1 for s in similarities if s and s < 0.7) if similarities else 0
            }
        }


# Global metrics collector instance
_metrics_collector: Optional[RetrievalMetricsCollector] = None


def get_metrics_collector() -> RetrievalMetricsCollector:
    """Get or create global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = RetrievalMetricsCollector()
    return _metrics_collector
