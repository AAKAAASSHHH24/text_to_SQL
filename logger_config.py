import os
import sys
from loguru import logger
from datetime import datetime
import json

class LoggerConfig:
    """Centralized logging configuration for the Text-to-SQL application"""
    
    def __init__(self):
        self.setup_logger()
    
    def setup_logger(self):
        """Configure loguru logger with multiple handlers and formats"""
        # Remove default handler
        logger.remove()
        
        # Console handler with colored output
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO",
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # File handler for all logs
        logger.add(
            "logs/app.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        # Error-specific file handler
        logger.add(
            "logs/errors.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation="5 MB",
            retention="60 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        # Performance logs
        logger.add(
            "logs/performance.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            level="INFO",
            filter=lambda record: "PERFORMANCE" in record["message"],
            rotation="5 MB",
            retention="30 days"
        )
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        logger.info("Logger initialized successfully")
    
    @staticmethod
    def log_function_entry(func_name: str, **kwargs):
        """Log function entry with parameters"""
        params = {k: str(v)[:100] + "..." if len(str(v)) > 100 else v for k, v in kwargs.items()}
        logger.debug(f"ENTRY: {func_name} called with params: {params}")
    
    @staticmethod
    def log_function_exit(func_name: str, execution_time: float , result_info: str):
        """Log function exit with timing and result info"""
        msg = f"EXIT: {func_name} completed"
        if execution_time:
            msg += f" in {execution_time:.3f}s"
        if result_info:
            msg += f" - {result_info}"
        logger.debug(msg)
    
    @staticmethod
    def log_performance(operation: str, duration: float, metadata: dict):
        """Log performance metrics"""
        perf_data = {
            "operation": operation,
            "duration_ms": round(duration * 1000, 2),
            "timestamp": datetime.now().isoformat()
        }
        if metadata:
            perf_data.update(metadata)
        logger.info(f"PERFORMANCE: {json.dumps(perf_data)}")

# Initialize logger
log_config = LoggerConfig()