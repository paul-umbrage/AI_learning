"""
Error handling and resilience utilities.

Provides retry logic, fallback strategies, and circuit breakers
for robust API interactions.
"""

import time
import random
from typing import Callable, Any, Optional, Type, Tuple
from functools import wraps
from enum import Enum


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker pattern for external API calls"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args, **kwargs: Function arguments
        
        Returns:
            Function result
        
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                # Try to recover
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception(f"Circuit breaker is OPEN. Service unavailable.")
        
        try:
            result = func(*args, **kwargs)
            # Success - reset failure count
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
            self.failure_count = 0
            return result
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
            
            raise


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to prevent thundering herd
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # Last attempt failed
                        raise
                    
                    # Calculate delay with exponential backoff
                    if jitter:
                        # Add random jitter (0-25% of delay)
                        jitter_amount = delay * 0.25 * random.random()
                        actual_delay = delay + jitter_amount
                    else:
                        actual_delay = delay
                    
                    actual_delay = min(actual_delay, max_delay)
                    
                    time.sleep(actual_delay)
                    delay *= exponential_base
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


class FallbackStrategy:
    """Manages fallback strategies for failed operations"""
    
    def __init__(self):
        self.strategies = []
    
    def add_fallback(self, strategy: Callable, condition: Optional[Callable] = None):
        """
        Add a fallback strategy
        
        Args:
            strategy: Fallback function to call
            condition: Optional condition function (returns bool)
        """
        self.strategies.append({
            'strategy': strategy,
            'condition': condition
        })
    
    def execute_with_fallback(self, primary_func: Callable, *args, **kwargs) -> Any:
        """
        Execute primary function with fallback strategies
        
        Args:
            primary_func: Primary function to try first
            *args, **kwargs: Arguments for functions
        
        Returns:
            Result from primary or first successful fallback
        
        Raises:
            Exception: If all strategies fail
        """
        last_exception = None
        
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            # Try fallback strategies
            for fallback in self.strategies:
                try:
                    # Check condition if provided
                    if fallback['condition'] and not fallback['condition'](e):
                        continue
                    
                    result = fallback['strategy'](*args, **kwargs)
                    return result
                except Exception as fallback_error:
                    last_exception = fallback_error
                    continue
            
            # All strategies failed
            raise last_exception


def graceful_degradation(default_value: Any = None):
    """
    Decorator for graceful degradation - returns default value on error
    
    Args:
        default_value: Value to return on error
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception:
                return default_value
        return wrapper
    return decorator


# Pre-configured circuit breakers for common services
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

