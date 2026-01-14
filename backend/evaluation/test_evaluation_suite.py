"""
Comprehensive evaluation test suite for RAG system.

This module provides extensive test cases covering:
- Edge cases in retrieval
- Different query types
- Boundary conditions
- Error scenarios
- Performance metrics
"""

import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluation.evaluator import RAGEvaluator
from database import search_similar_chunks
from reranking import rerank_chunks
from context_assembly import assemble_context, deduplicate_chunks
from hybrid_search import hybrid_search


class EdgeCaseDocumentation:
    """Document observed edge cases and their handling."""
    
    EDGE_CASES = {
        "empty_query": {
            "description": "Empty or very short queries",
            "observed_behavior": "System returns empty context, handled gracefully",
            "handling": "Query validation in build_rag_context returns early with empty results",
            "status": "handled"
        },
        "no_results": {
            "description": "Queries that return no matching chunks",
            "observed_behavior": "System returns empty context without error",
            "handling": "Warning logged, empty context returned",
            "status": "handled"
        },
        "very_long_query": {
            "description": "Queries exceeding typical length (1000+ characters)",
            "observed_behavior": "Embedding generation may be slower, but works",
            "handling": "No explicit truncation, relies on embedding model limits",
            "status": "monitored"
        },
        "duplicate_chunks": {
            "description": "Multiple chunks with identical or near-identical content",
            "observed_behavior": "Deduplication removes exact duplicates",
            "handling": "deduplicate_chunks function uses hash-based and similarity-based deduplication",
            "status": "handled"
        },
        "low_similarity_results": {
            "description": "All retrieved chunks have low similarity scores (<0.5)",
            "observed_behavior": "Results still returned but may not be relevant",
            "handling": "min_similarity threshold (0.7) filters low-quality results",
            "status": "handled"
        },
        "context_window_overflow": {
            "description": "Retrieved chunks exceed token limit",
            "observed_behavior": "Chunks truncated at sentence boundaries",
            "handling": "assemble_context truncates intelligently, reserves tokens for prompt",
            "status": "handled"
        },
        "special_characters": {
            "description": "Queries with special characters, emojis, or unicode",
            "observed_behavior": "Embedding model handles most cases",
            "handling": "No explicit filtering, relies on embedding model",
            "status": "monitored"
        },
        "multi_language": {
            "description": "Queries in languages other than English",
            "observed_behavior": "Works if embedding model supports the language",
            "handling": "No explicit language detection, relies on model capabilities",
            "status": "monitored"
        },
        "numeric_queries": {
            "description": "Queries asking for numbers, dates, or statistics",
            "observed_behavior": "May require exact matches which semantic search may miss",
            "handling": "Hybrid search with keyword matching helps",
            "status": "handled"
        },
        "ambiguous_queries": {
            "description": "Queries with multiple possible interpretations",
            "observed_behavior": "May return diverse results",
            "handling": "Query expansion can help, but may need manual disambiguation",
            "status": "partial"
        },
        "very_short_chunks": {
            "description": "Chunks with less than 30 characters",
            "observed_behavior": "May lack context, penalized in reranking",
            "handling": "Length penalty in rerank_by_content_relevance",
            "status": "handled"
        },
        "very_long_chunks": {
            "description": "Chunks exceeding 2000 characters",
            "observed_behavior": "May be truncated or penalized",
            "handling": "Length filtering in rerank_by_length_penalty",
            "status": "handled"
        },
        "concurrent_requests": {
            "description": "Multiple simultaneous requests",
            "observed_behavior": "Rate limiting prevents abuse",
            "handling": "RateLimitMiddleware with per-IP and per-endpoint limits",
            "status": "handled"
        },
        "token_limit_exceeded": {
            "description": "Daily token limit exceeded",
            "observed_behavior": "Requests rejected with 429 status",
            "handling": "Token tracking in RateLimitMiddleware",
            "status": "handled"
        }
    }
    
    @classmethod
    def get_all_cases(cls) -> Dict[str, Dict[str, str]]:
        """Get all documented edge cases."""
        return cls.EDGE_CASES
    
    @classmethod
    def get_case(cls, case_name: str) -> Optional[Dict[str, str]]:
        """Get a specific edge case."""
        return cls.EDGE_CASES.get(case_name)
    
    @classmethod
    def document_new_case(cls, case_name: str, description: str, behavior: str, handling: str, status: str = "observed"):
        """Document a new edge case."""
        cls.EDGE_CASES[case_name] = {
            "description": description,
            "observed_behavior": behavior,
            "handling": handling,
            "status": status
        }


def create_test_suite() -> List[Dict[str, Any]]:
    """
    Create comprehensive test suite covering various scenarios.
    
    Returns:
        List of test case dictionaries
    """
    test_cases = [
        # Basic retrieval tests
        {
            "name": "simple_definition_query",
            "query": "What is machine learning?",
            "expected_chunks": [
                "machine learning",
                "artificial intelligence",
                "algorithms"
            ],
            "description": "Basic definition query"
        },
        {
            "name": "list_query",
            "query": "List types of neural networks",
            "expected_chunks": [
                "neural network",
                "types",
                "varieties"
            ],
            "description": "Query asking for a list"
        },
        {
            "name": "howto_query",
            "query": "How does backpropagation work?",
            "expected_chunks": [
                "backpropagation",
                "gradient",
                "training"
            ],
            "description": "How-to/process query"
        },
        
        # Edge case tests
        {
            "name": "very_short_query",
            "query": "AI",
            "expected_chunks": ["artificial intelligence"],
            "description": "Very short query (2 characters)"
        },
        {
            "name": "very_long_query",
            "query": "What is machine learning and how does it differ from traditional programming approaches in terms of data handling and algorithm development?",
            "expected_chunks": ["machine learning", "programming"],
            "description": "Very long query (100+ characters)"
        },
        {
            "name": "numeric_query",
            "query": "What is the learning rate 0.01 used for?",
            "expected_chunks": ["learning rate", "0.01"],
            "description": "Query with numeric value"
        },
        {
            "name": "special_characters",
            "query": "What is RAG (Retrieval-Augmented Generation)?",
            "expected_chunks": ["RAG", "retrieval", "generation"],
            "description": "Query with special characters and acronyms"
        },
        {
            "name": "comparison_query",
            "query": "Compare supervised and unsupervised learning",
            "expected_chunks": ["supervised", "unsupervised", "learning"],
            "description": "Comparison query"
        },
        {
            "name": "ambiguous_query",
            "query": "What is a model?",
            "expected_chunks": [],  # Could be ML model, statistical model, etc.
            "description": "Ambiguous query (multiple meanings)"
        },
        
        # Boundary tests
        {
            "name": "empty_query",
            "query": "",
            "expected_chunks": [],
            "description": "Empty query string"
        },
        {
            "name": "whitespace_only",
            "query": "   ",
            "expected_chunks": [],
            "description": "Query with only whitespace"
        },
        {
            "name": "single_word",
            "query": "learning",
            "expected_chunks": ["learning"],
            "description": "Single word query"
        },
        {
            "name": "question_mark_only",
            "query": "?",
            "expected_chunks": [],
            "description": "Query with only punctuation"
        },
        
        # Complex queries
        {
            "name": "multi_part_query",
            "query": "Explain neural networks and their applications in computer vision",
            "expected_chunks": ["neural network", "computer vision", "applications"],
            "description": "Multi-part query with multiple concepts"
        },
        {
            "name": "negation_query",
            "query": "What is not machine learning?",
            "expected_chunks": ["machine learning"],
            "description": "Query with negation"
        },
        {
            "name": "temporal_query",
            "query": "What are recent developments in AI?",
            "expected_chunks": ["AI", "developments", "recent"],
            "description": "Temporal query"
        }
    ]
    
    return test_cases


def run_evaluation_suite(
    evaluator: RAGEvaluator,
    test_cases: Optional[List[Dict[str, Any]]] = None,
    use_reranking: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation suite.
    
    Args:
        evaluator: RAGEvaluator instance
        test_cases: Optional custom test cases (uses default if None)
        use_reranking: Whether to use reranking
        verbose: Print detailed results
    
    Returns:
        Comprehensive evaluation results
    """
    if test_cases is None:
        test_cases = create_test_suite()
    
    if verbose:
        print("=" * 80)
        print("RAG System Evaluation Suite")
        print("=" * 80)
        print(f"Total test cases: {len(test_cases)}")
        print(f"Reranking enabled: {use_reranking}")
        print("=" * 80)
    
    # Run evaluation
    results = evaluator.run_evaluation_suite(
        test_cases=test_cases,
        use_reranking=use_reranking
    )
    
    if verbose:
        print("\n" + "=" * 80)
        print("Evaluation Results Summary")
        print("=" * 80)
        print(f"Test Cases Run: {results.get('num_test_cases', 0)}")
        print(f"Average Precision: {results.get('average_precision', 0):.3f}")
        print(f"Average Recall: {results.get('average_recall', 0):.3f}")
        print(f"Average F1 Score: {results.get('average_f1', 0):.3f}")
        print(f"Average MRR: {results.get('average_mrr', 0):.3f}")
        print(f"Average NDCG: {results.get('average_ndcg', 0):.3f}")
        print(f"Average Similarity: {results.get('average_similarity', 0):.3f}")
        print(f"Average Retrieval Time: {results.get('average_retrieval_time_ms', 0):.2f} ms")
        
        if 'precision_at_k' in results:
            print("\nPrecision@K:")
            for k, value in results['precision_at_k'].items():
                print(f"  {k}: {value:.3f}")
        
        print("=" * 80)
    
    return results


def test_edge_cases(evaluator: RAGEvaluator, verbose: bool = True) -> Dict[str, Any]:
    """
    Test specific edge cases.
    
    Args:
        evaluator: RAGEvaluator instance
        verbose: Print detailed results
    
    Returns:
        Edge case test results
    """
    edge_case_tests = [
        {
            "name": "empty_query_edge_case",
            "query": "",
            "expected_chunks": [],
            "description": "Edge case: Empty query"
        },
        {
            "name": "very_short_query_edge_case",
            "query": "A",
            "expected_chunks": [],
            "description": "Edge case: Single character query"
        },
        {
            "name": "special_chars_edge_case",
            "query": "What is @#$%?",
            "expected_chunks": [],
            "description": "Edge case: Special characters only"
        },
        {
            "name": "unicode_edge_case",
            "query": "什么是机器学习？",  # "What is machine learning?" in Chinese
            "expected_chunks": [],
            "description": "Edge case: Non-English query"
        }
    ]
    
    if verbose:
        print("\n" + "=" * 80)
        print("Edge Case Testing")
        print("=" * 80)
    
    results = {}
    for test_case in edge_case_tests:
        try:
            metrics = evaluator.evaluate_retrieval(
                query=test_case["query"],
                expected_chunks=test_case["expected_chunks"],
                use_reranking=True,
                top_k=5
            )
            results[test_case["name"]] = {
                "status": "completed",
                "metrics": metrics,
                "description": test_case["description"]
            }
            if verbose:
                print(f"\n{test_case['name']}: {test_case['description']}")
                print(f"  Status: {'✅ Passed' if 'error' not in metrics else '❌ Failed'}")
        except Exception as e:
            results[test_case["name"]] = {
                "status": "error",
                "error": str(e),
                "description": test_case["description"]
            }
            if verbose:
                print(f"\n{test_case['name']}: {test_case['description']}")
                print(f"  Status: ❌ Error - {e}")
    
    return results


def generate_evaluation_report(
    suite_results: Dict[str, Any],
    edge_case_results: Dict[str, Any],
    output_file: Optional[str] = None
) -> str:
    """
    Generate comprehensive evaluation report.
    
    Args:
        suite_results: Results from evaluation suite
        edge_case_results: Results from edge case tests
        output_file: Optional file path to save report
    
    Returns:
        Report as string
    """
    report_lines = [
        "=" * 80,
        "RAG System Evaluation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
        "SUMMARY METRICS",
        "-" * 80,
        f"Test Cases Run: {suite_results.get('num_test_cases', 0)}",
        f"Average Precision: {suite_results.get('average_precision', 0):.3f}",
        f"Average Recall: {suite_results.get('average_recall', 0):.3f}",
        f"Average F1 Score: {suite_results.get('average_f1', 0):.3f}",
        f"Average MRR: {suite_results.get('average_mrr', 0):.3f}",
        f"Average NDCG: {suite_results.get('average_ndcg', 0):.3f}",
        f"Average Similarity: {suite_results.get('average_similarity', 0):.3f}",
        f"Average Retrieval Time: {suite_results.get('average_retrieval_time_ms', 0):.2f} ms",
        "",
        "EDGE CASES DOCUMENTED",
        "-" * 80,
    ]
    
    # Add edge case documentation
    edge_cases = EdgeCaseDocumentation.get_all_cases()
    for case_name, case_info in edge_cases.items():
        report_lines.append(f"{case_name}:")
        report_lines.append(f"  Description: {case_info['description']}")
        report_lines.append(f"  Status: {case_info['status']}")
        report_lines.append("")
    
    report_lines.extend([
        "EDGE CASE TEST RESULTS",
        "-" * 80,
    ])
    
    for case_name, case_result in edge_case_results.items():
        report_lines.append(f"{case_name}:")
        report_lines.append(f"  Status: {case_result.get('status', 'unknown')}")
        if 'metrics' in case_result:
            metrics = case_result['metrics']
            report_lines.append(f"  Precision: {metrics.get('precision', 0):.3f}")
            report_lines.append(f"  Recall: {metrics.get('recall', 0):.3f}")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    
    report = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")
    
    return report


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Run full evaluation suite
    print("Running comprehensive evaluation suite...")
    suite_results = run_evaluation_suite(evaluator, verbose=True)
    
    # Test edge cases
    print("\nTesting edge cases...")
    edge_case_results = test_edge_cases(evaluator, verbose=True)
    
    # Generate report
    print("\nGenerating evaluation report...")
    report = generate_evaluation_report(
        suite_results,
        edge_case_results,
        output_file="evaluation_report.txt"
    )
    
    print("\n" + report)
