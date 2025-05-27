import time
import functools
from typing import Any, Callable
from loguru import logger

def log_execution(include_args: bool = True, include_result: bool = False):
    """Decorator to log function execution with timing"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            # Log entry
            if include_args:
                args_str = str(args)[:200] + "..." if len(str(args)) > 200 else str(args)
                kwargs_str = str(kwargs)[:200] + "..." if len(str(kwargs)) > 200 else str(kwargs)
                logger.debug(f"ENTRY: {func_name} | args={args_str} | kwargs={kwargs_str}")
            else:
                logger.debug(f"ENTRY: {func_name}")
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log successful exit
                result_info = ""
                if include_result and result is not None:
                    result_str = str(result)
                    result_info = f" | result={result_str[:100] + '...' if len(result_str) > 100 else result_str}"
                
                logger.debug(f"EXIT: {func_name} | duration={execution_time:.3f}s{result_info}")
                
                # Log performance metrics
                if execution_time > 0.1:  # Log slow operations
                    logger.info(f"PERFORMANCE: {func_name} took {execution_time:.3f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"ERROR: {func_name} failed after {execution_time:.3f}s | error={str(e)}")
                raise
        
        return wrapper
    return decorator

def log_errors(custom_message: str):
    """Decorator to log errors with context"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                func_name = f"{func.__module__}.{func.__qualname__}"
                error_msg = custom_message or f"Error in {func_name}"
                logger.error(f"{error_msg}: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator