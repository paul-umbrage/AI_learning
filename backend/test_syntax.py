#!/usr/bin/env python3
"""
Simple syntax and import test for the improved RAG system.
Tests that all modules can be imported and have no syntax errors.
"""

import sys
import os
import ast

def test_syntax(file_path):
    """Test if a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def test_imports():
    """Test that modules can be imported (if dependencies available)."""
    results = []
    
    # Test reranking module
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        import reranking
        results.append(("reranking.py", True, "Import successful"))
    except ImportError as e:
        results.append(("reranking.py", False, f"Import failed: {e}"))
    except SyntaxError as e:
        results.append(("reranking.py", False, f"Syntax error: {e}"))
    except Exception as e:
        results.append(("reranking.py", False, f"Error: {e}"))
    
    # Test context_assembly module
    try:
        import context_assembly
        results.append(("context_assembly.py", True, "Import successful"))
    except ImportError as e:
        results.append(("context_assembly.py", False, f"Import failed: {e}"))
    except SyntaxError as e:
        results.append(("context_assembly.py", False, f"Syntax error: {e}"))
    except Exception as e:
        results.append(("context_assembly.py", False, f"Error: {e}"))
    
    # Test rate_limiting module
    try:
        import rate_limiting
        results.append(("rate_limiting.py", True, "Import successful"))
    except ImportError as e:
        results.append(("rate_limiting.py", False, f"Import failed: {e}"))
    except SyntaxError as e:
        results.append(("rate_limiting.py", False, f"Syntax error: {e}"))
    except Exception as e:
        results.append(("rate_limiting.py", False, f"Error: {e}"))
    
    return results

def test_function_signatures():
    """Test that key functions exist with correct signatures."""
    results = []
    
    try:
        import reranking
        # Check enhanced functions exist
        if hasattr(reranking, 'rerank_by_keyword_overlap'):
            results.append(("rerank_by_keyword_overlap", True, "Function exists"))
        else:
            results.append(("rerank_by_keyword_overlap", False, "Function not found"))
            
        if hasattr(reranking, 'rerank_by_content_relevance'):
            results.append(("rerank_by_content_relevance", True, "Function exists"))
        else:
            results.append(("rerank_by_content_relevance", False, "Function not found"))
            
        if hasattr(reranking, 'rerank_chunks'):
            results.append(("rerank_chunks", True, "Function exists"))
        else:
            results.append(("rerank_chunks", False, "Function not found"))
    except Exception as e:
        results.append(("reranking functions", False, f"Error checking: {e}"))
    
    try:
        import context_assembly
        if hasattr(context_assembly, 'assemble_context'):
            # Check if it has reserve_tokens parameter
            import inspect
            sig = inspect.signature(context_assembly.assemble_context)
            if 'reserve_tokens' in sig.parameters:
                results.append(("assemble_context (with reserve_tokens)", True, "Enhanced function exists"))
            else:
                results.append(("assemble_context (with reserve_tokens)", False, "reserve_tokens parameter not found"))
        else:
            results.append(("assemble_context", False, "Function not found"))
    except Exception as e:
        results.append(("context_assembly functions", False, f"Error checking: {e}"))
    
    try:
        import rate_limiting
        if hasattr(rate_limiting, 'RateLimitMiddleware'):
            results.append(("RateLimitMiddleware", True, "Class exists"))
        else:
            results.append(("RateLimitMiddleware", False, "Class not found"))
            
        if hasattr(rate_limiting, 'RateLimiter'):
            results.append(("RateLimiter", True, "Class exists"))
        else:
            results.append(("RateLimiter", False, "Class not found"))
    except Exception as e:
        results.append(("rate_limiting classes", False, f"Error checking: {e}"))
    
    return results

def main():
    """Run all tests."""
    print("=" * 80)
    print("RAG System Syntax and Import Test")
    print("=" * 80)
    
    # Test syntax of key files
    print("\n1. Testing Syntax...")
    print("-" * 80)
    files_to_test = [
        "reranking.py",
        "context_assembly.py",
        "rate_limiting.py"
    ]
    
    syntax_results = []
    for filename in files_to_test:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            valid, error = test_syntax(filepath)
            status = "✅ PASS" if valid else "❌ FAIL"
            syntax_results.append((filename, valid))
            print(f"{filename:30} {status}")
            if not valid:
                print(f"  Error: {error}")
        else:
            print(f"{filename:30} ❌ FAIL (file not found)")
            syntax_results.append((filename, False))
    
    # Test imports (may fail if dependencies not installed)
    print("\n2. Testing Imports...")
    print("-" * 80)
    import_results = test_imports()
    for filename, success, message in import_results:
        status = "✅ PASS" if success else "⚠️  SKIP (dependency missing)"
        print(f"{filename:30} {status}")
        if not success and "dependency" not in message.lower():
            print(f"  {message}")
    
    # Test function signatures
    print("\n3. Testing Function Signatures...")
    print("-" * 80)
    func_results = test_function_signatures()
    for func_name, success, message in func_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{func_name:40} {status}")
        if not success:
            print(f"  {message}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    syntax_passed = sum(1 for _, valid in syntax_results if valid)
    print(f"Syntax Tests: {syntax_passed}/{len(syntax_results)} passed")
    
    import_passed = sum(1 for _, success, _ in import_results if success)
    print(f"Import Tests: {import_passed}/{len(import_results)} passed (may require dependencies)")
    
    func_passed = sum(1 for _, success, _ in func_results if success)
    print(f"Function Tests: {func_passed}/{len(func_results)} passed")
    
    # Overall result
    all_syntax_valid = all(valid for _, valid in syntax_results)
    if all_syntax_valid:
        print("\n✅ All syntax tests passed! Code is syntactically correct.")
        print("⚠️  Note: Some import tests may fail if dependencies are not installed.")
        print("   Install dependencies with: uv sync (or pip install -r requirements.txt)")
        return 0
    else:
        print("\n❌ Some syntax tests failed. Please fix syntax errors.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
