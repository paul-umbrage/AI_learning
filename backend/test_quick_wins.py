#!/usr/bin/env python3
"""
Test script to verify quick wins implementation
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_logger():
    """Test structured logging"""
    print("Testing structured logging...")
    try:
        from utils.logger import get_logger, calculate_token_cost
        
        logger = get_logger("test")
        
        # Test basic logging
        logger.info("test_event", test_field="test_value")
        print("  ✅ Basic logging works")
        
        # Test request logging
        logger.log_request("POST", "/api/test", "test-123", {"test": "data"})
        print("  ✅ Request logging works")
        
        # Test cost calculation
        cost = calculate_token_cost("gpt-3.5-turbo", 100, 50)
        print(f"  ✅ Cost calculation works: ${cost:.4f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Logger test failed: {e}")
        return False

def test_prompt_templates():
    """Test prompt template system"""
    print("\nTesting prompt templates...")
    try:
        from prompts.templates import build_rag_prompt, PromptStrategy
        from prompts.strategies import select_strategy, get_strategy_from_preset, list_strategies
        
        # Test strategy selection
        strategy = select_strategy("What is machine learning?")
        print(f"  ✅ Strategy selection works: {strategy.value}")
        
        # Test preset
        preset_strategy = get_strategy_from_preset("technical")
        print(f"  ✅ Preset selection works: {preset_strategy.value}")
        
        # Test prompt building
        system, user = build_rag_prompt(
            query="What is RAG?",
            context="RAG is a technique...",
            strategy=PromptStrategy.TECHNICAL,
            use_rag=True
        )
        assert len(system) > 0
        assert len(user) > 0
        assert "RAG" in user
        print("  ✅ Prompt building works")
        
        # Test strategy listing
        strategies = list_strategies()
        assert len(strategies) > 0
        print(f"  ✅ Strategy listing works: {len(strategies)} strategies")
        
        return True
    except Exception as e:
        print(f"  ❌ Prompt template test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_metrics():
    """Test evaluation metrics"""
    print("\nTesting evaluation metrics...")
    try:
        from evaluation.evaluator import RAGEvaluator
        
        evaluator = RAGEvaluator()
        
        # Test metric calculation with mock data
        retrieved = [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "Random text that doesn't match"
        ]
        expected = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks"
        ]
        similarities = [0.9, 0.85, 0.3]
        
        metrics = evaluator._calculate_retrieval_metrics(retrieved, expected, similarities)
        
        # Check that new metrics exist
        assert "mrr" in metrics
        assert "ndcg" in metrics
        assert "precision_at_k" in metrics
        print(f"  ✅ MRR calculation works: {metrics['mrr']}")
        print(f"  ✅ NDCG calculation works: {metrics['ndcg']}")
        print(f"  ✅ Precision@K works: {metrics['precision_at_k']}")
        
        return True
    except Exception as e:
        print(f"  ❌ Evaluation metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_imports():
    """Test that main.py imports work"""
    print("\nTesting main.py imports...")
    try:
        from main import app, ChatRequest, ChatResponse
        print("  ✅ Main app imports successfully")
        print("  ✅ ChatRequest model works")
        print("  ✅ ChatResponse model works")
        
        # Test ChatRequest with new prompt_strategy field
        request = ChatRequest(
            message="Test",
            prompt_strategy="technical"
        )
        assert request.prompt_strategy == "technical"
        print("  ✅ Prompt strategy parameter works")
        
        return True
    except Exception as e:
        print(f"  ❌ Main import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Quick Wins Implementation")
    print("=" * 60)
    
    results = []
    results.append(("Logger", test_logger()))
    results.append(("Prompt Templates", test_prompt_templates()))
    results.append(("Evaluation Metrics", test_evaluation_metrics()))
    results.append(("Main Imports", test_main_imports()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:25} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed! Quick wins are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

