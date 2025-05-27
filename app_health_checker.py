
import os
import sqlite3
import time
from datetime import datetime
from typing import Dict, Any
from loguru import logger
from decorators import log_execution

class HealthChecker:
    """Monitor application health and dependencies"""
    
    @staticmethod
    @log_execution()
    def check_database_health(db_path: str) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        logger.info("Checking database health")
        
        health_status = {
            "status": "unknown",
            "response_time": None,
            "error": None,
            "table_count": 0,
            "file_size": 0
        }
        
        start_time = time.time()
        try:
            
            # Check file exists
            if not os.path.exists(db_path):
                health_status["status"] = "error"
                health_status["error"] = "Database file not found"
                return health_status
            
            # Check file size
            health_status["file_size"] = os.path.getsize(db_path)
            
            # Test connection and query
            with sqlite3.connect(db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                health_status["table_count"] = cursor.fetchone()[0]
            
            health_status["response_time"] = time.time() - start_time
            health_status["status"] = "healthy"
            
            logger.success(f"Database health check passed - {health_status['table_count']} tables, {health_status['response_time']:.3f}s response time")
            
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = str(e)
            health_status["response_time"] = time.time() - start_time
            logger.error(f"Database health check failed: {str(e)}")
        
        return health_status
    
    @staticmethod
    @log_execution()
    def check_llm_health(llm) -> Dict[str, Any]:
        """Check LLM connectivity and response"""
        logger.info("Checking LLM health")
        
        health_status = {
            "status": "unknown",
            "response_time": None,
            "error": None,
            "test_response": None
        }
        start_time = time.time()
        
        try:
            
            # Simple test query
            test_response = llm.invoke("Respond with 'OK' if you receive this message.")
            
            health_status["response_time"] = time.time() - start_time
            health_status["test_response"] = str(test_response)[:100]
            health_status["status"] = "healthy"
            
            logger.success(f"LLM health check passed - {health_status['response_time']:.3f}s response time")
            
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = str(e)
            health_status["response_time"] = time.time() - start_time
            logger.error(f"LLM health check failed: {str(e)}")
        
        return health_status