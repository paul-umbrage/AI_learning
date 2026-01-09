"""
Simple evaluation framework for RAG system.

This module provides basic evaluation metrics to measure retrieval
and response quality.
"""

from typing import List, Dict, Any, Tuple, Optional
import time
from openai import OpenAI
import os
from dotenv import load_dotenv
from database import search_similar_chunks
from reranking import rerank_chunks

load_dotenv()

def get_query_embedding(query: str) -> list:
    """Generate embedding for a query string"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    
    client = OpenAI(api_key=api_key)
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


class RAGEvaluator:
    """Evaluate RAG system performance."""
    
    def __init__(self, api_key: str = None):
        """Initialize evaluator."""
        self.api_key = api_key
        self.metrics_history = []
    
    def evaluate_retrieval(
        self,
        query: str,
        expected_chunks: List[str],
        use_reranking: bool = True,
        top_k: int = 5,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval quality.
        
        Args:
            query: Test query
            expected_chunks: List of chunk texts that should be retrieved
            use_reranking: Whether to use reranking
            top_k: Number of chunks to retrieve
            filename: Optional filename filter
        
        Returns:
            Dictionary with evaluation metrics
        """
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = get_query_embedding(query)
        if not query_embedding:
            return {"error": "Failed to generate query embedding"}
        
        # Retrieve chunks
        initial_limit = top_k * 2 if use_reranking else top_k
        results = search_similar_chunks(
            query_embedding, 
            filename=filename, 
            limit=initial_limit
        )
        
        # Apply reranking if enabled
        if use_reranking:
            results = rerank_chunks(
                query=query,
                results=results,
                strategy="combined",
                top_k=top_k,
                min_similarity=0.7
            )
        else:
            results = results[:top_k]
        
        retrieval_time = time.time() - start_time
        
        # Extract retrieved chunk texts
        retrieved_chunks = [r[0] for r in results]
        retrieved_similarities = [r[3] for r in results]
        
        # Calculate metrics
        metrics = self._calculate_retrieval_metrics(
            retrieved_chunks,
            expected_chunks,
            retrieved_similarities
        )
        
        metrics["retrieval_time_ms"] = retrieval_time * 1000
        metrics["chunks_retrieved"] = len(retrieved_chunks)
        metrics["use_reranking"] = use_reranking
        
        return metrics
    
    def _calculate_retrieval_metrics(
        self,
        retrieved: List[str],
        expected: List[str],
        similarities: List[float]
    ) -> Dict[str, Any]:
        """Calculate precision, recall, MRR, NDCG and other metrics."""
        
        # Simple text-based matching (for evaluation purposes)
        # In production, you'd use more sophisticated matching
        
        retrieved_lower = [chunk.lower() for chunk in retrieved]
        expected_lower = [chunk.lower() for chunk in expected]
        
        # Calculate relevance scores for each retrieved chunk
        relevance_scores = []
        matches = 0
        
        for i, retrieved_chunk in enumerate(retrieved_lower):
            is_relevant = False
            for expected_chunk in expected_lower:
                # Simple overlap check (can be improved)
                if len(expected_chunk) > 0 and len(retrieved_chunk) > 0:
                    # Check if significant portion matches
                    words_expected = set(expected_chunk.split()[:10])  # First 10 words
                    words_retrieved = set(retrieved_chunk.split()[:10])
                    if words_expected and words_retrieved:
                        overlap = len(words_expected & words_retrieved) / len(words_expected)
                        if overlap > 0.3:  # 30% word overlap
                            is_relevant = True
                            matches += 1
                            break
            
            # Relevance score: 1 if relevant, 0 otherwise
            relevance_scores.append(1 if is_relevant else 0)
        
        # Precision: relevant retrieved / total retrieved
        precision = matches / len(retrieved) if retrieved else 0.0
        
        # Recall: relevant retrieved / total relevant
        recall = matches / len(expected) if expected else 0.0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Precision@K for different K values
        precision_at_k = {}
        for k in [1, 3, 5, 10]:
            if k <= len(relevance_scores):
                precision_at_k[f"precision@{k}"] = round(
                    sum(relevance_scores[:k]) / k, 3
                )
        
        # Mean Reciprocal Rank (MRR)
        # Find the rank of the first relevant document
        first_relevant_rank = None
        for i, score in enumerate(relevance_scores, start=1):
            if score > 0:
                first_relevant_rank = i
                break
        
        mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        
        # Normalized Discounted Cumulative Gain (NDCG)
        # DCG = sum(relevance_i / log2(i+1)) for i in ranks
        dcg = sum(
            rel / (__import__('math').log2(i + 2))  # i+2 because log2(1) = 0
            for i, rel in enumerate(relevance_scores)
        )
        
        # Ideal DCG (if all relevant docs were at the top)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = sum(
            rel / (__import__('math').log2(i + 2))
            for i, rel in enumerate(ideal_relevance)
        )
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        # Mean similarity score
        mean_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        # Min similarity (worst match)
        min_similarity = min(similarities) if similarities else 0.0
        
        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "matches": matches,
            "mrr": round(mrr, 3),
            "ndcg": round(ndcg, 3),
            "precision_at_k": precision_at_k,
            "mean_similarity": round(mean_similarity, 3),
            "min_similarity": round(min_similarity, 3),
            "max_similarity": round(max(similarities), 3) if similarities else 0.0
        }
    
    def evaluate_response_quality(
        self,
        query: str,
        response: str,
        sources: List[Dict[str, Any]],
        expected_answer: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate response quality.
        
        Args:
            query: Original query
            response: Generated response
            sources: Source chunks used
            expected_answer: Expected answer (optional)
        
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            "response_length": len(response),
            "num_sources": len(sources),
            "has_sources": len(sources) > 0
        }
        
        # Check if response mentions sources
        if sources:
            source_filenames = [s.get("filename", "") for s in sources]
            metrics["source_filenames"] = source_filenames
        
        # If expected answer provided, calculate similarity
        if expected_answer:
            # Simple word overlap (can be improved with embeddings)
            response_words = set(response.lower().split())
            expected_words = set(expected_answer.lower().split())
            
            if expected_words:
                word_overlap = len(response_words & expected_words) / len(expected_words)
                metrics["answer_overlap"] = round(word_overlap, 3)
            else:
                metrics["answer_overlap"] = 0.0
        
        return metrics
    
    def run_evaluation_suite(
        self,
        test_cases: List[Dict[str, Any]],
        use_reranking: bool = True
    ) -> Dict[str, Any]:
        """
        Run evaluation on multiple test cases.
        
        Args:
            test_cases: List of test cases, each with:
                - "query": Test query
                - "expected_chunks": List of expected chunk texts
                - "filename": Optional filename filter
            use_reranking: Whether to use reranking
        
        Returns:
            Aggregated evaluation results
        """
        all_metrics = []
        
        for i, test_case in enumerate(test_cases):
            query = test_case["query"]
            expected_chunks = test_case.get("expected_chunks", [])
            filename = test_case.get("filename")
            
            print(f"Evaluating test case {i+1}/{len(test_cases)}: {query[:50]}...")
            
            metrics = self.evaluate_retrieval(
                query=query,
                expected_chunks=expected_chunks,
                use_reranking=use_reranking,
                filename=filename
            )
            
            metrics["query"] = query
            all_metrics.append(metrics)
        
        # Aggregate metrics
        if all_metrics:
            avg_precision = sum(m.get("precision", 0) for m in all_metrics) / len(all_metrics)
            avg_recall = sum(m.get("recall", 0) for m in all_metrics) / len(all_metrics)
            avg_f1 = sum(m.get("f1_score", 0) for m in all_metrics) / len(all_metrics)
            avg_mrr = sum(m.get("mrr", 0) for m in all_metrics) / len(all_metrics)
            avg_ndcg = sum(m.get("ndcg", 0) for m in all_metrics) / len(all_metrics)
            avg_similarity = sum(m.get("mean_similarity", 0) for m in all_metrics) / len(all_metrics)
            avg_retrieval_time = sum(m.get("retrieval_time_ms", 0) for m in all_metrics) / len(all_metrics)
            
            # Aggregate precision@K
            precision_at_k_agg = {}
            for k in [1, 3, 5, 10]:
                key = f"precision@{k}"
                values = [m.get("precision_at_k", {}).get(key, 0) for m in all_metrics]
                if values:
                    precision_at_k_agg[f"avg_{key}"] = round(sum(values) / len(values), 3)
            
            return {
                "num_test_cases": len(test_cases),
                "average_precision": round(avg_precision, 3),
                "average_recall": round(avg_recall, 3),
                "average_f1": round(avg_f1, 3),
                "average_mrr": round(avg_mrr, 3),
                "average_ndcg": round(avg_ndcg, 3),
                "average_similarity": round(avg_similarity, 3),
                "average_retrieval_time_ms": round(avg_retrieval_time, 2),
                "precision_at_k": precision_at_k_agg,
                "detailed_results": all_metrics
            }
        
        return {"error": "No metrics calculated"}


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    # Note: This requires get_query_embedding to be accessible
    # You may need to import it properly or make it a function parameter
    
    evaluator = RAGEvaluator()
    
    # Example test cases
    test_cases = [
        {
            "query": "What is machine learning?",
            "expected_chunks": [
                "Machine learning is a subset of artificial intelligence",
                "Machine learning algorithms learn from data"
            ]
        },
        {
            "query": "How does RAG work?",
            "expected_chunks": [
                "RAG combines retrieval with generation",
                "RAG retrieves relevant context from documents"
            ]
        }
    ]
    
    print("Running evaluation suite...")
    results = evaluator.run_evaluation_suite(test_cases, use_reranking=True)
    print("\nEvaluation Results:")
    print(f"Average Precision: {results.get('average_precision', 0)}")
    print(f"Average Recall: {results.get('average_recall', 0)}")
    print(f"Average F1: {results.get('average_f1', 0)}")

