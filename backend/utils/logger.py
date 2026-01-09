"""
Structured logging utilities for the RAG system.

Provides JSON-formatted logging with request/response tracking,
timing information, and error context.
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from functools import wraps
import traceback


class StructuredLogger:
    """Structured logger that outputs JSON-formatted logs"""
    
    def __init__(self, name: str = "rag_system", log_file: Optional[str] = None):
        """
        Initialize structured logger
        
        Args:
            name: Logger name
            log_file: Optional file path to write logs to
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create formatter for JSON output
        self.json_formatter = JSONFormatter()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.json_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self.json_formatter)
            self.logger.addHandler(file_handler)
    
    def _log(self, level: str, event: str, data: Dict[str, Any]):
        """Internal method to log structured data"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.upper(),
            "event": event,
            **data
        }
        
        if level == "info":
            self.logger.info(json.dumps(log_entry))
        elif level == "warning":
            self.logger.warning(json.dumps(log_entry))
        elif level == "error":
            self.logger.error(json.dumps(log_entry))
        elif level == "debug":
            self.logger.debug(json.dumps(log_entry))
    
    def info(self, event: str, **kwargs):
        """Log info level event"""
        self._log("info", event, kwargs)
    
    def warning(self, event: str, **kwargs):
        """Log warning level event"""
        self._log("warning", event, kwargs)
    
    def error(self, event: str, error: Optional[Exception] = None, **kwargs):
        """Log error level event"""
        error_data = kwargs.copy()
        if error:
            error_data.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc()
            })
        self._log("error", event, error_data)
    
    def debug(self, event: str, **kwargs):
        """Log debug level event"""
        self._log("debug", event, kwargs)
    
    def log_request(self, method: str, path: str, request_id: str, 
                   body: Optional[Dict[str, Any]] = None):
        """Log incoming API request"""
        self.info(
            "api_request",
            request_id=request_id,
            method=method,
            path=path,
            body=body
        )
    
    def log_response(self, request_id: str, status_code: int, 
                    response_time_ms: float, response_size: Optional[int] = None):
        """Log API response"""
        self.info(
            "api_response",
            request_id=request_id,
            status_code=status_code,
            response_time_ms=round(response_time_ms, 2),
            response_size=response_size
        )
    
    def log_embedding_generation(self, request_id: str, query: str, 
                                 model: str, duration_ms: float, 
                                 embedding_dim: Optional[int] = None):
        """Log embedding generation"""
        self.info(
            "embedding_generated",
            request_id=request_id,
            query_preview=query[:100] if len(query) > 100 else query,
            model=model,
            duration_ms=round(duration_ms, 2),
            embedding_dimension=embedding_dim
        )
    
    def log_retrieval(self, request_id: str, query: str, top_k: int,
                     chunks_retrieved: int, retrieval_time_ms: float,
                     use_reranking: bool = False, rerank_strategy: Optional[str] = None,
                     avg_similarity: Optional[float] = None):
        """Log RAG retrieval operation"""
        self.info(
            "rag_retrieval",
            request_id=request_id,
            query_preview=query[:100] if len(query) > 100 else query,
            top_k=top_k,
            chunks_retrieved=chunks_retrieved,
            retrieval_time_ms=round(retrieval_time_ms, 2),
            use_reranking=use_reranking,
            rerank_strategy=rerank_strategy,
            avg_similarity=round(avg_similarity, 3) if avg_similarity else None
        )
    
    def log_llm_call(self, request_id: str, model: str, prompt_tokens: int,
                    completion_tokens: int, total_tokens: int,
                    duration_ms: float, cost_usd: Optional[float] = None):
        """Log LLM API call"""
        self.info(
            "llm_call",
            request_id=request_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_ms=round(duration_ms, 2),
            estimated_cost_usd=round(cost_usd, 4) if cost_usd else None
        )
    
    def log_function_call(self, request_id: str, function_name: str,
                         arguments: Dict[str, Any], result: Any,
                         duration_ms: float):
        """Log function call execution"""
        self.info(
            "function_call",
            request_id=request_id,
            function_name=function_name,
            arguments=arguments,
            result_preview=str(result)[:200] if result else None,
            duration_ms=round(duration_ms, 2)
        )


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs JSON"""
    
    def format(self, record):
        # If message is already JSON, return as-is
        if isinstance(record.msg, str):
            try:
                json.loads(record.msg)
                return record.msg
            except:
                pass
        
        # Otherwise format as JSON
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(log_data)


# Global logger instance
_logger_instance: Optional[StructuredLogger] = None


def get_logger(name: str = "rag_system", log_file: Optional[str] = None) -> StructuredLogger:
    """Get or create global logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = StructuredLogger(name, log_file)
    return _logger_instance


def timed_log(logger: StructuredLogger, event_name: str):
    """Decorator to log function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            request_id = kwargs.get('request_id', 'unknown')
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    event_name,
                    request_id=request_id,
                    function=func.__name__,
                    duration_ms=round(duration_ms, 2),
                    status="success"
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"{event_name}_error",
                    request_id=request_id,
                    function=func.__name__,
                    duration_ms=round(duration_ms, 2),
                    error=e
                )
                raise
        return wrapper
    return decorator


def calculate_token_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate estimated cost in USD for OpenAI API call
    
    Pricing as of 2024 (approximate):
    - gpt-3.5-turbo: $0.0015/1K prompt, $0.002/1K completion
    - gpt-4: $0.03/1K prompt, $0.06/1K completion
    - text-embedding-ada-002: $0.0001/1K tokens
    """
    pricing = {
        "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
        "text-embedding-ada-002": {"prompt": 0.0001, "completion": 0.0001}
    }
    
    model_pricing = pricing.get(model, pricing["gpt-3.5-turbo"])
    prompt_cost = (prompt_tokens / 1000) * model_pricing["prompt"]
    completion_cost = (completion_tokens / 1000) * model_pricing["completion"]
    
    return prompt_cost + completion_cost

