
import sqlite3
import pandas as pd
from typing import Optional, Dict, List
from loguru import logger
from decorators import log_execution, log_errors
import time
import os

class DatabaseHandler:
    """Enhanced database handler with comprehensive logging"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        logger.info(f"Initializing DatabaseHandler with path: {db_path}")
        self.validate_database()
    
    @log_execution()
    @log_errors("Database validation failed")
    def validate_database(self):
        """Validate database exists and is accessible"""
        if not os.path.exists(self.db_path):
            logger.error(f"Database file does not exist: {self.db_path}")
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
                logger.info(f"Database validation successful. Found {table_count} tables.")
        except Exception as e:
            logger.error(f"Database validation failed: {str(e)}")
            raise
    
    @log_execution(include_args=True)
    @log_errors("Database query execution failed")
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame with detailed logging"""
        logger.info(f"Executing query: {query[:100]}...")
        
        start_time = time.time()
        
        try:
            # Log query analysis
            query_upper = query.upper()
            query_type = "SELECT" if "SELECT" in query_upper else "OTHER"
            has_joins = "JOIN" in query_upper
            has_aggregation = any(agg in query_upper for agg in ["GROUP BY", "COUNT", "SUM", "AVG", "MAX", "MIN"])
            
            logger.debug(f"Query analysis - Type: {query_type}, Joins: {has_joins}, Aggregation: {has_aggregation}")
            
            with sqlite3.connect(self.db_path) as conn:
                logger.debug("Database connection established")
                df = pd.read_sql_query(query, conn)
                
            execution_time = time.time() - start_time
            
            logger.success(f"Query executed successfully in {execution_time:.3f}s")
            logger.info(f"Result: {len(df)} rows, {len(df.columns)} columns")
            
            if not df.empty:
                logger.debug(f"Columns: {list(df.columns)}")
                logger.debug(f"Data types: {df.dtypes.to_dict()}")
                
                # Log memory usage
                memory_usage = df.memory_usage(deep=True).sum()
                logger.debug(f"Result DataFrame memory usage: {memory_usage} bytes")
            
            return df
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query execution failed after {execution_time:.3f}s: {str(e)}")
            logger.error(f"Failed query: {query}")
            raise Exception(f"Database query failed: {str(e)}")
    
    @log_execution()
    @log_errors("Failed to get table information")
    def get_table_info(self) -> Dict[str, List[str]]:
        """Get information about all tables with logging"""
        logger.info("Retrieving database table information")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                logger.info(f"Found {len(tables)} tables")
                
                table_info = {}
                for table in tables:
                    table_name = table[0]
                    logger.debug(f"Getting info for table: {table_name}")
                    
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = cursor.fetchall()
                    column_names = [col[1] for col in columns]
                    
                    table_info[table_name] = column_names
                    logger.debug(f"Table {table_name} has {len(column_names)} columns: {column_names}")
                
                logger.success(f"Retrieved information for {len(table_info)} tables")
                return table_info
                
        except Exception as e:
            logger.error(f"Failed to get table info: {str(e)}")
            raise Exception(f"Failed to get table info: {str(e)}")
    
    @log_execution(include_args=True)
    def get_sample_data(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """Get sample data from a table with logging"""
        logger.info(f"Getting sample data from table: {table_name}, limit: {limit}")
        
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            df = self.execute_query(query)
            logger.success(f"Retrieved {len(df)} sample rows from {table_name}")
            return df
        except Exception as e:
            logger.error(f"Failed to get sample data from {table_name}: {str(e)}")
            raise