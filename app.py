import streamlit as st
import pandas as pd
import sqlite3
import os
import time
import traceback
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

import plotly.express as px
import plotly.graph_objects as go
import re
import json
from typing import Optional, Tuple, Dict, Any

from loguru import logger
from decorators import log_execution, log_errors
from logger_config import LoggerConfig

# Load environment variables
load_dotenv()

class SQLCaptureHandler(BaseCallbackHandler):
    """Callback handler to capture the SQL string from the agent's tool invocation."""
    # def __init__(self):
    #     super().__init__()
    #     self.captured_sql: Optional[str] = None

    # def on_tool_start(self, serialized, input_str: str, **kwargs):
    #     # Assuming the agent uses a tool named 'execute_sql' or similar
    #     if serialized.get("name", "").lower().startswith("execute_sql"):
    #         self.captured_sql = input_str

    def __init__(self):
        super().__init__()
        self.intermediate_steps = []

    def on_agent_action(self, action, **kwargs):
        # Record each agent action (the SQL query should be one of these actions)
        self.intermediate_steps.append(("action", action))

    def on_agent_finish(self, finish, **kwargs):
        # Optionally, record when the agent finishes processing.
        self.intermediate_steps.append(("finish", finish))


class TextToSQLApp:
    """Main Text-to-SQL Application with comprehensive logging"""
    
    def __init__(self):
        logger.info("Initializing Text-to-SQL Application")
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Session ID: {self.session_id}")
        
        try:
            self.setup_page_config()
            self.initialize_database()
            self.initialize_llm()
            self.setup_sql_agent()
            logger.success("Application initialized successfully")
        except Exception as e:
            logger.critical(f"Application initialization failed: {str(e)}", exc_info=True)
            raise
    
    @log_execution()
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        logger.debug("Setting up Streamlit page configuration")
        
        config = {
            "page_title": "Text-to-SQL Assistant",
            "page_icon": "🔍",
        }
        
        st.set_page_config(
            page_title="Text-to-SQL Assistant",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        logger.info(f"Page configuration set: {config}")
    
    @log_execution()
    @log_errors("Database initialization failed")
    def initialize_database(self):
        """Initialize SQLite database connection with logging"""
        logger.info("Initializing database connection")
        
        db_path = "chinook.db"
        logger.debug(f"Database path: {db_path}")
        
        if not os.path.exists(db_path):
            logger.error(f"Database file not found: {db_path}")
            st.error("Database file 'chinook.db' not found. Please ensure the Chinook database is in the root directory.")
            st.stop()
        
        file_size = os.path.getsize(db_path)
        logger.info(f"Database file size: {file_size} bytes")
        
        try:
            self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
            
            # Test connection
            test_query = "SELECT COUNT(*) as table_count FROM sqlite_master WHERE type='table'"
            result = self.db.run(test_query)
            logger.info(f"Database connection test successful. Tables found: {result}")
            
            # Log table information
            self.log_database_schema()
            
            st.success("Database connected successfully!")
            logger.success("Database initialization completed")
            
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}", exc_info=True)
            st.error(f"Database connection failed: {str(e)}")
            st.stop()
    
    @log_execution()
    def log_database_schema(self):
        """Log database schema information"""
        try:
            tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
            tables_result = self.db.run(tables_query)
            logger.info(f"Database tables: {tables_result}")
            
            # Log schema for each table
            table_names = re.findall(r"'(\w+)'", str(tables_result))
            for table in table_names:
                try:
                    schema_query = f"PRAGMA table_info({table})"
                    schema_result = self.db.run(schema_query)
                    logger.debug(f"Schema for table {table}: {schema_result}")
                except Exception as e:
                    logger.warning(f"Could not get schema for table {table}: {str(e)}")
                    
        except Exception as e:
            logger.warning(f"Could not log database schema: {str(e)}")
    
    @log_execution()
    @log_errors("LLM initialization failed")
    def initialize_llm(self):
        """Initialize Groq LLM with logging"""
        logger.info("Initializing Groq LLM")
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.error("GROQ_API_KEY not found in environment variables")
            st.error("GROQ_API_KEY not found in environment variables")
            st.stop()
        
        # Mask API key for logging
        masked_key = f"{groq_api_key[:8]}***{groq_api_key[-4:]}" if len(groq_api_key) > 12 else "***"
        logger.debug(f"Using Groq API key: {masked_key}")
        
        llm_config = {
            "temperature": 0,
            "model_name": "llama-3.3-70b-versatile",
            "groq_api_key": groq_api_key
        }
        
        try:
            start_time = time.time()
            self.llm = ChatGroq(**llm_config)
            
            # Test LLM with a simple query
            test_response = self.llm.invoke("Hello, respond with 'OK'")
            init_time = time.time() - start_time
            
            logger.info(f"LLM initialization successful in {init_time:.3f}s")
            logger.debug(f"LLM test response: {test_response}")
            logger.info(f"Using model: {llm_config['model_name']}")
            
        except Exception as e:
            logger.error(f"LLM initialization failed: {str(e)}", exc_info=True)
            st.error(f"LLM initialization failed: {str(e)}")
            st.stop()
    
    @log_execution()
    @log_errors("SQL agent setup failed")
    def setup_sql_agent(self):
        """Setup SQL agent with custom prompt and logging"""
        logger.info("Setting up SQL agent")
        
        try:
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            logger.debug("SQL toolkit created successfully")
            
            custom_prompt = PromptTemplate(
                template="""
                You are a SQL expert assistant. Given a user question, create a SQL query to answer it.
                
                Database Schema Information:
                - albums: AlbumId, Title, ArtistId
                - artists: ArtistId, Name
                - customers: CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId
                - employees: EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email
                - genres: GenreId, Name
                - invoice_items: InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity
                - invoices: InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total
                - media_types: MediaTypeId, Name
                - playlist_track: PlaylistId, TrackId
                - playlists: PlaylistId, Name
                - tracks: TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice
                
                Important guidelines:
                1. Always use proper JOIN statements when relating tables
                2. Use LIMIT clauses for queries that might return many rows
                3. Format currency values appropriately
                4. Return results that would be suitable for visualization when applicable
                5. Be precise with column names and table relationships
                
                User Question: {input}
                
                Create a SQL query to answer this question. Only return the SQL query, no explanation.
                """,
                input_variables=["input"]
            )
            
            logger.debug("Creating SQL agent with custom prompt")
            self.sql_agent = create_sql_agent(
                llm=self.llm,
                toolkit=toolkit,
                verbose=True,
                handle_parsing_errors=True
                # ,prompt=custom_prompt
            )
            
            logger.success("SQL agent setup completed successfully")
            
        except Exception as e:
            logger.error(f"SQL agent setup failed: {str(e)}", exc_info=True)
            raise
    
    @log_execution(include_args=True, include_result=True)
    def extract_sql_query(self, agent_response: str) -> Optional[str]:
        """Extract SQL query from agent response with detailed logging"""
        logger.debug(f"Extracting SQL query from agent response (length: {len(str(agent_response))})")
        
        response_text = str(agent_response)
        logger.debug(f"Agent response preview: {response_text[:200]}...")
        
        # SQL extraction patterns
        sql_patterns = [
            r'```sql\n(.*?)\n```',
            r'```\n(SELECT.*?)\n```',
            r'(SELECT.*?)(?:\n|$)',
        ]
        
        for i, pattern in enumerate(sql_patterns):
            logger.debug(f"Trying SQL pattern {i+1}: {pattern}")
            matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if matches:
                extracted_sql = matches[0].strip()
                logger.info(f"SQL extracted using pattern {i+1}: {extracted_sql[:100]}...")
                return extracted_sql
        
        # Fallback: look for SELECT statements
        if 'SELECT' in response_text.upper():
            logger.debug("Trying fallback SELECT extraction")
            lines = response_text.split('\n')
            sql_lines = []
            in_sql = False
            for line in lines:
                if 'SELECT' in line.upper():
                    in_sql = True
                if in_sql:
                    sql_lines.append(line)
                    if ';' in line:
                        break
            
            if sql_lines:
                extracted_sql = '\n'.join(sql_lines).strip()
                logger.info(f"SQL extracted using fallback method: {extracted_sql[:100]}...")
                return extracted_sql
        
        logger.warning("No SQL query could be extracted from agent response")
        return None
    
    @log_execution(include_args=True)
    def determine_chart_type(self, query: str, data: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Determine if and what type of chart to generate with logging"""
        logger.info(f"Determining chart type for query: '{query[:50]}...' with data shape: {data.shape}")
        
        if data.empty:
            logger.warning("Data is empty, no chart will be generated")
            return None, None, None
        
        if len(data.columns) < 2:
            logger.warning(f"Data has only {len(data.columns)} columns, need at least 2 for charting")
            return None, None, None
        
        query_lower = query.lower()
        logger.debug(f"Query lowercase: {query_lower}")
        
        # Keywords for different chart types
        temporal_keywords = ['time', 'date', 'year', 'month', 'day', 'trend', 'over time']
        comparison_keywords = ['compare', 'vs', 'versus', 'top', 'bottom', 'most', 'least', 'ranking']
        distribution_keywords = ['distribution', 'percentage', 'proportion', 'share', 'breakdown']
        
        # Check query context
        has_temporal = any(keyword in query_lower for keyword in temporal_keywords)
        has_comparison = any(keyword in query_lower for keyword in comparison_keywords)
        has_distribution = any(keyword in query_lower for keyword in distribution_keywords)
        
        logger.debug(f"Query analysis - temporal: {has_temporal}, comparison: {has_comparison}, distribution: {has_distribution}")
        
        # Analyze data structure
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        logger.debug(f"Data analysis - numeric columns: {numeric_cols}, categorical columns: {categorical_cols}")
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found, cannot create chart")
            return None, None, None
        
        x_col = categorical_cols[0] if categorical_cols else data.columns[0]
        y_col = numeric_cols[0]
        
        logger.debug(f"Selected columns - X: {x_col}, Y: {y_col}")
        
        # Determine chart type based on context and data
        chart_type = None
        if has_temporal or 'date' in str(data.columns).lower():
            chart_type = 'line'
            logger.info("Selected line chart due to temporal context")
        elif has_distribution and len(data) <= 10:
            chart_type = 'pie'
            logger.info("Selected pie chart due to distribution context and small dataset")
        elif has_comparison or len(categorical_cols) > 0:
            chart_type = 'bar'
            logger.info("Selected bar chart due to comparison context or categorical data")
        elif len(data) <= 20:
            chart_type = 'bar'
            logger.info("Selected bar chart due to small dataset size")
        else:
            chart_type = 'line'
            logger.info("Selected line chart as default for larger dataset")
        
        logger.info(f"Chart decision: type={chart_type}, x_col={x_col}, y_col={y_col}")
        return chart_type, x_col, y_col
    
    @log_execution(include_args=True)
    @log_errors("Chart creation failed")
    def create_chart(self, chart_type: str, data: pd.DataFrame, x_col: str, y_col: str):
        """Create chart based on type and data with logging"""
        logger.info(f"Creating {chart_type} chart with x_col='{x_col}', y_col='{y_col}'")
        
        start_time = time.time()
        
        try:
            if chart_type == 'bar':
                logger.debug("Creating bar chart")
                fig = px.bar(data, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            elif chart_type == 'pie':
                logger.debug("Creating pie chart")
                fig = px.pie(data, names=x_col, values=y_col, title=f"{y_col} Distribution")
            elif chart_type == 'line':
                logger.debug("Creating line chart")
                fig = px.line(data, x=x_col, y=y_col, title=f"{y_col} Trend")
                fig.update_traces(mode='lines+markers')
            else:
                logger.error(f"Unknown chart type: {chart_type}")
                return None
            
            fig.update_layout(height=400)
            
            creation_time = time.time() - start_time
            logger.success(f"{chart_type.title()} chart created successfully in {creation_time:.3f}s")
            
            # Log performance metrics
            LoggerConfig.log_performance(
                operation=f"create_{chart_type}_chart",
                duration=creation_time,
                metadata={"data_rows": len(data), "x_col": x_col, "y_col": y_col}
            )
            
            return fig
        
        except Exception as e:
            creation_time = time.time() - start_time
            logger.error(f"Chart creation failed after {creation_time:.3f}s: {str(e)}", exc_info=True)
            return None
    
    @log_execution(include_args=True)
    @log_errors("Query execution failed")
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query and return results with logging"""
        logger.info(f"Executing SQL query: {sql_query[:100]}...")
        
        start_time = time.time()
        
        try:
            # Log query details
            query_lines = sql_query.strip().split('\n')
            logger.debug(f"Query has {len(query_lines)} lines")
            
            # Estimate query complexity
            complexity_indicators = ['JOIN', 'GROUP BY', 'ORDER BY', 'HAVING', 'SUBQUERY']
            complexity_score = sum(1 for indicator in complexity_indicators if indicator in sql_query.upper())
            logger.debug(f"Query complexity score: {complexity_score}")
            
            conn = sqlite3.connect("chinook.db")
            logger.debug("Database connection established for query execution")
            
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            execution_time = time.time() - start_time
            
            logger.success(f"Query executed successfully in {execution_time:.3f}s")
            logger.info(f"Query returned {len(df)} rows and {len(df.columns)} columns")
            
            if len(df) > 0:
                logger.debug(f"Column names: {list(df.columns)}")
                logger.debug(f"Data types: {df.dtypes.to_dict()}")
                
                # Log sample data (first row)
                if len(df) > 0:
                    sample_row = df.iloc[0].to_dict()
                    logger.debug(f"Sample row: {sample_row}")
            
            # Log performance metrics
            LoggerConfig.log_performance(
                operation="sql_query_execution",
                duration=execution_time,
                metadata={
                    "rows_returned": len(df),
                    "columns_returned": len(df.columns),
                    "complexity_score": complexity_score,
                    "query_length": len(sql_query)
                }
            )
            
            return df
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query execution failed after {execution_time:.3f}s: {str(e)}", exc_info=True)
            logger.error(f"Failed query: {sql_query}")
            return pd.DataFrame()
    
    @log_execution(include_args=True)
    def process_natural_language_query(self, user_query: str) -> Tuple:
        """Process natural language query and return results with comprehensive logging"""
        logger.info(f"Processing natural language query: '{user_query}'")
        
        # Input validation and sanitization
        if not isinstance(user_query, str) or len(user_query.strip()) == 0:
            logger.warning("User query is empty or not a string")
            st.warning("Please enter a valid question.")
            return None, None, None, None, None
        if len(user_query) > 1000:
            logger.warning("User query too long")
            st.warning("Your question is too long. Please shorten it.")
            return None, None, None, None, None
        # Basic sanitization: remove dangerous SQL keywords (defense-in-depth)
        forbidden_patterns = [";--", "--", "/*", "*/", "DROP", "DELETE", "INSERT", "UPDATE"]
        if any(pat.lower() in user_query.lower() for pat in forbidden_patterns):
            logger.warning("Potentially dangerous input detected in user query")
            st.warning("Your question contains potentially unsafe content.")
            return None, None, None, None, None
        
        start_time = time.time()
        query_id = f"query_{int(time.time())}"
        
        logger.info(f"Query ID: {query_id}")
        
        try:
            handler = SQLCaptureHandler()
            logger.info("Step 1: Converting natural language to SQL using LLM (with capture)")
            llm_start = time.time()
            with st.spinner("Converting natural language to SQL..."):
                # Pass our callback handler to capture the SQL before execution
                agent_output = self.sql_agent.run(
                    user_query,
                    callbacks=[handler]
                )
            llm_time = time.time() - llm_start
            logger.info(f"c in {llm_time:.3f}s")

            # Retrieve SQL directly from callback
                # Look through logged events to capture the SQL query
            sql_query = None
            for event_type, event in handler.intermediate_steps:
                if event_type == "action" and getattr(event, "tool", None) == 'sql_db_query':
                    sql_query = getattr(event, "tool_input", None)
                    
            result,sql_query = agent_output, sql_query
            logger.info(f"Agent output: {result}...")  # Log first 100 chars of agent output
            
            #sql_query = handler.captured_sql
            if not sql_query:
                logger.error("Failed to capture SQL from agent workflow.")
                return None, None, None, None, None

            logger.success(f"Captured SQL query: {sql_query}...")

            # Step 2: Query Execution
            logger.info("Step 2: Executing SQL query")
            with st.spinner("Executing SQL query..."):
                df = self.execute_query(sql_query)

            if df.empty:
                logger.warning("Query returned empty results")
                total_time = time.time() - start_time
                logger.info(f"Total processing time: {total_time:.3f}s")
                return sql_query, df, None, None, None

            # Step 3: Determine if chart is needed
            logger.info("Step 3: Determining chart generation requirements")
            chart_type, x_col, y_col = self.determine_chart_type(user_query, df)

            # Step 4: Chart Creation
            chart = None
            if chart_type:
                logger.info(f"Step 4: Creating {chart_type} chart")
                with st.spinner(f"Generating {chart_type} chart..."):
                    chart = self.create_chart(chart_type, df, x_col, y_col)
            else:
                logger.info("Step 4: No chart generation required")

            total_time = time.time() - start_time
            logger.success(f"Query processing completed successfully in {total_time:.3f}s")
            logger.info(f"Results summary - Rows: {len(df)}, Columns: {len(df.columns)}, Chart: {chart_type or 'None'}")

            # Log performance metrics
            LoggerConfig.log_performance(
                operation="full_query_processing",
                duration=total_time,
                metadata={
                    "query_id": query_id,
                    "user_query_length": len(user_query),
                    "sql_query_length": len(sql_query),
                    "result_rows": len(df),
                    "result_columns": len(df.columns),
                    "chart_type": chart_type or "none",
                    "llm_time": llm_time
                }
            )

            return sql_query, df, chart, chart_type, (x_col, y_col) if chart_type else None
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Query processing failed after {total_time:.3f}s: {str(e)}", exc_info=True)
            logger.error(f"Failed query: {user_query}")
            
            # Log error metrics
            LoggerConfig.log_performance(
                operation="failed_query_processing",
                duration=total_time,
                metadata={
                    "query_id": query_id,
                    "user_query": user_query,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            
            return None, None, None, None, None
    
    @log_execution()
    def display_sample_queries(self):
        """Display sample queries in sidebar with logging"""
        logger.debug("Displaying sample queries in sidebar")
        
        st.sidebar.header("📋 Sample Queries")
        
        sample_queries = [
            "Show me the top 10 best-selling tracks",
            "What are the total sales by country?",
            "List all rock genre albums",
            "Show monthly sales trends for 2021",
            "Which artists have the most albums?",
            "What is the average track length by genre?",
            "Show customer distribution by country",
            "List the top 5 customers by total purchases",
            "What are the most popular media types?",
            "Show sales performance by employee"
        ]
        
        logger.debug(f"Displaying {len(sample_queries)} sample queries")
        
        for i, query in enumerate(sample_queries, 1):
            if st.sidebar.button(f"{i}. {query}", key=f"sample_{i}"):
                logger.info(f"Sample query selected: {query}")
                st.session_state.selected_query = query
    
    @log_execution()
    def display_logs_sidebar(self):
        """Display recent logs in sidebar for debugging"""
        if st.sidebar.checkbox("Show Debug Logs"):
            logger.debug("Displaying debug logs in sidebar")
            
            try:
                with open("logs/app.log", "r") as f:
                    logs = f.readlines()
                    recent_logs = logs[-20:]  # Last 20 log entries
                    
                st.sidebar.subheader("Recent Logs")
                for log in recent_logs:
                    if "ERROR" in log:
                        st.sidebar.error(log.strip())
                    elif "WARNING" in log:
                        st.sidebar.warning(log.strip())
                    elif "SUCCESS" in log:
                        st.sidebar.success(log.strip())
                    else:
                        st.sidebar.text(log.strip())
                        
            except Exception as e:
                logger.warning(f"Could not display logs: {str(e)}")
                st.sidebar.error("Could not load debug logs")
    
    @log_execution()
    def run(self):
        """Main application runner with comprehensive logging"""
        logger.info("Starting main application runner")

        try:
            st.title("🔍 Text-to-SQL Assistant")
            st.markdown("Convert natural language questions into SQL queries and visualize results")

            # Display session info
            st.sidebar.markdown(f"**Session ID:** `{self.session_id}`")
            st.sidebar.markdown(f"**Start Time:** {datetime.now().strftime('%H:%M:%S')}")

            # --- Sidebar with sample queries and logs ---
            # Use session state to track sample query selection and trigger
            if "selected_query" not in st.session_state:
                st.session_state.selected_query = ""
            if "trigger_process" not in st.session_state:
                st.session_state.trigger_process = False

            # Sample queries logic
            st.sidebar.header("📋 Sample Queries")
            sample_queries = [
                "Show me the top 10 best-selling tracks",
                "What are the total sales by country?",
                "List all rock genre albums",
                "Show monthly sales trends for 2021",
                "Which artists have the most albums?",
                "What is the average track length by genre?",
                "Show customer distribution by country",
                "List the top 5 customers by total purchases",
                "What are the most popular media types?",
                "Show sales performance by employee"
            ]
            for i, query in enumerate(sample_queries, 1):
                if st.sidebar.button(f"{i}. {query}", key=f"sample_{i}"):
                    logger.info(f"Sample query selected: {query}")
                    st.session_state.selected_query = query
                    st.session_state.trigger_process = False  # Reset trigger

            # Logs sidebar
            self.display_logs_sidebar()

            # --- Main interface ---
            col1, col2 = st.columns([3, 1])

            with col1:
                # Query input
                user_query = st.text_area(
                    "Enter your question:",
                    value=st.session_state.selected_query,
                    height=100,
                    placeholder="e.g., Show me the top 10 best-selling tracks",
                    key="user_query_input"
                )

            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                process_button = st.button("🚀 Process Query", type="primary", key="process_query_btn")

            # --- Query execution logic ---
            # If process button is clicked, or if a sample query was selected and process button is clicked
            # We use a trigger flag to ensure that after selecting a sample query, clicking the process button executes it
            if process_button:
                st.session_state.trigger_process = True

            if st.session_state.trigger_process and user_query.strip():
                logger.info(f"Processing button clicked with query: '{user_query.strip()}'")

                try:
                    sql_query, df, chart, chart_type, chart_cols = self.process_natural_language_query(user_query)
                    logger.info(f"Natural language Query processing completed: SQL query : {sql_query}, DataFrame: {df}, Chart info: {chart}, Chart type: {chart_type}, Chart columns: {chart_cols}")

                    if sql_query and not df.empty:
                        logger.info("Displaying successful results")

                        # Display SQL query
                        st.subheader("📝 Generated SQL Query")
                        st.code(sql_query, language="sql")
                        logger.debug("SQL query displayed in UI")

                        # Display results
                        if chart:
                            col1, col2 = st.columns([1, 1])
                        else:
                            col1 = st.columns([1])[0]
                        with col1:
                            st.subheader("📊 Query Results")
                            st.dataframe(df, use_container_width=True)
                            st.caption(f"Showing {len(df)} rows")
                            logger.debug("Results table displayed in UI")

                        if chart:
                            with col2:
                                st.subheader(f"📈 {chart_type.title()} Chart")
                                st.plotly_chart(chart, use_container_width=True)
                                if chart_cols:
                                    st.caption(f"X-axis: {chart_cols[0]}, Y-axis: {chart_cols[1]}")
                                logger.debug(f"Chart displayed in UI: {chart_type}")

                    elif sql_query:
                        logger.info("Query successful but no results returned")
                        st.subheader("📝 Generated SQL Query")
                        st.code(sql_query, language="sql")
                        st.info("Query executed successfully but returned no results.")

                    else:
                        logger.error("Query processing failed completely")
                        st.error("Failed to process your query. Please check the logs for details.")

                except Exception as e:
                    logger.error(f"Unexpected error in query processing: {str(e)}", exc_info=True)
                    st.error(f"An unexpected error occurred: {str(e)}")

                    # Display error details in expander for debugging
                    with st.expander("Error Details (for debugging)"):
                        st.code(traceback.format_exc())

                # After processing, reset trigger so it doesn't auto-run on next rerun
                st.session_state.trigger_process = False

            elif process_button and not user_query.strip():
                logger.warning("Process button clicked but no query provided")
                st.warning("Please enter a question to process.")

            # --- Database info ---
            with st.expander("ℹ️ Database Information"):
                st.markdown("""
                **Chinook Database Schema:**
                - **artists**: ArtistId, Name
                - **albums**: AlbumId, Title, ArtistId  
                - **tracks**: TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice
                - **customers**: CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId
                - **invoices**: InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total
                - **invoice_items**: InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity
                - **employees**: EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email
                - **genres**: GenreId, Name
                - **media_types**: MediaTypeId, Name
                - **playlists**: PlaylistId, Name
                - **playlist_track**: PlaylistId, TrackId
                """)

            # --- Performance metrics (for admins) ---
            if st.checkbox("Show Performance Metrics"):
                self.display_performance_metrics()

            logger.info("Main application runner completed successfully")

        except Exception as e:
            logger.critical(f"Critical error in main application runner: {str(e)}", exc_info=True)
            st.error("A critical error occurred. Please check the logs.")
            raise
    
    
    
    # def run(self):
    #     """Main application runner with comprehensive logging"""
    #     logger.info("Starting main application runner")
        
    #     try:
    #         st.title("🔍 Text-to-SQL Assistant")
    #         st.markdown("Convert natural language questions into SQL queries and visualize results")
            
    #         # Display session info
    #         st.sidebar.markdown(f"**Session ID:** `{self.session_id}`")
    #         st.sidebar.markdown(f"**Start Time:** {datetime.now().strftime('%H:%M:%S')}")
            
    #         # Sidebar with sample queries and logs
    #         self.display_sample_queries()
    #         self.display_logs_sidebar()
            
    #         # Main interface
    #         col1, col2 = st.columns([3, 1])
            
    #         with col1:
    #             # Query input
    #             if 'selected_query' in st.session_state:
    #                 default_query = st.session_state.selected_query
    #                 del st.session_state.selected_query
    #                 logger.info(f"Using selected query: {default_query}")
    #             else:
    #                 default_query = ""
                
    #             user_query = st.text_area(
    #                 "Enter your question:",
    #                 value=default_query,
    #                 height=100,
    #                 placeholder="e.g., Show me the top 10 best-selling tracks"
    #             )
            
    #         with col2:
    #             st.markdown("<br>", unsafe_allow_html=True)
    #             process_button = st.button("🚀 Process Query", type="primary")
            
    #         # Process query
    #         if process_button and user_query.strip():
    #             logger.info(f"Processing button clicked with query: '{user_query.strip()}'")
                
    #             try:
    #                 sql_query, df, chart, chart_type, chart_cols = self.process_natural_language_query(user_query)
    #                 logger.info(f"Natural language Query processing completed: SQL query : {sql_query}, DataFrame: {df}, Chart info: {chart}, Chart type: {chart_type}, Chart columns: {chart_cols}")
                    
    #                 if sql_query and not df.empty:
    #                     logger.info("Displaying successful results")
                        
    #                     # Display SQL query
    #                     st.subheader("📝 Generated SQL Query")
    #                     st.code(sql_query, language="sql")
    #                     logger.debug("SQL query displayed in UI")
                        
    #                     # Display results
    #                     if chart:
    #                         col1, col2 = st.columns([1, 1]) 
    #                     else: 
    #                         col1 = st.columns([1])[0]
    #                     with col1:
    #                         st.subheader("📊 Query Results")
    #                         st.dataframe(df, use_container_width=True)
    #                         st.caption(f"Showing {len(df)} rows")
    #                         logger.debug("Results table displayed in UI")
                        
    #                     if chart:
    #                         with col2:
    #                             st.subheader(f"📈 {chart_type.title()} Chart")
    #                             st.plotly_chart(chart, use_container_width=True)
    #                             if chart_cols:
    #                                 st.caption(f"X-axis: {chart_cols[0]}, Y-axis: {chart_cols[1]}")
    #                             logger.debug(f"Chart displayed in UI: {chart_type}")
                    
    #                 elif sql_query:
    #                     logger.info("Query successful but no results returned")
    #                     st.subheader("📝 Generated SQL Query")
    #                     st.code(sql_query, language="sql")
    #                     st.info("Query executed successfully but returned no results.")
                    
    #                 else:
    #                     logger.error("Query processing failed completely")
    #                     st.error("Failed to process your query. Please check the logs for details.")
                
    #             except Exception as e:
    #                 logger.error(f"Unexpected error in query processing: {str(e)}", exc_info=True)
    #                 st.error(f"An unexpected error occurred: {str(e)}")
                    
    #                 # Display error details in expander for debugging
    #                 with st.expander("Error Details (for debugging)"):
    #                     st.code(traceback.format_exc())
            
    #         elif process_button:
    #             logger.warning("Process button clicked but no query provided")
    #             st.warning("Please enter a question to process.")
            
    #         # Database info
    #         with st.expander("ℹ️ Database Information"):
    #             st.markdown("""
    #             **Chinook Database Schema:**
    #             - **artists**: ArtistId, Name
    #             - **albums**: AlbumId, Title, ArtistId  
    #             - **tracks**: TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice
    #             - **customers**: CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId
    #             - **invoices**: InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total
    #             - **invoice_items**: InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity
    #             - **employees**: EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email
    #             - **genres**: GenreId, Name
    #             - **media_types**: MediaTypeId, Name
    #             - **playlists**: PlaylistId, Name
    #             - **playlist_track**: PlaylistId, TrackId
    #             """)
            
    #         # Performance metrics (for admins)
    #         if st.checkbox("Show Performance Metrics"):
    #             self.display_performance_metrics()
                
    #         logger.info("Main application runner completed successfully")
            
    #     except Exception as e:
    #         logger.critical(f"Critical error in main application runner: {str(e)}", exc_info=True)
    #         st.error("A critical error occurred. Please check the logs.")
    #         raise
    
    @log_execution()
    def display_performance_metrics(self):
        """Display performance metrics from logs"""
        logger.debug("Displaying performance metrics")
        
        try:
            st.subheader("📈 Performance Metrics")
            
            # Read performance logs
            perf_data = []
            try:
                with open("logs/performance.log", "r") as f:
                    for line in f:
                        if "PERFORMANCE:" in line:
                            try:
                                json_part = line.split("PERFORMANCE: ")[1]
                                perf_entry = json.loads(json_part)
                                perf_data.append(perf_entry)
                            except (json.JSONDecodeError, IndexError):
                                continue
            except FileNotFoundError:
                st.info("No performance data available yet.")
                return
            
            if perf_data:
                df_perf = pd.DataFrame(perf_data)
                
                # Recent performance
                st.write("**Recent Operations:**")
                recent_perf = df_perf.tail(10)
                st.dataframe(recent_perf)
                
                # Performance summary
                if 'duration_ms' in df_perf.columns:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_duration = df_perf['duration_ms'].mean()
                        st.metric("Avg Duration", f"{avg_duration:.2f}ms")
                    
                    with col2:
                        max_duration = df_perf['duration_ms'].max()
                        st.metric("Max Duration", f"{max_duration:.2f}ms")
                    
                    with col3:
                        total_operations = len(df_perf)
                        st.metric("Total Operations", total_operations)
                
                # Performance chart
                if len(df_perf) > 1:
                    fig = px.line(df_perf.tail(20), 
                                y='duration_ms', 
                                title='Recent Operation Performance',
                                labels={'duration_ms': 'Duration (ms)'})
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("No performance metrics available.")
                
        except Exception as e:
            logger.error(f"Error displaying performance metrics: {str(e)}")
            st.error("Could not load performance metrics.")
            

if __name__ == "__main__":
    # Initialize logging configuration before any logger usage
    LoggerConfig()
    try:
        logger.info("="*50)
        logger.info("Starting Text-to-SQL Application")
        logger.info(f"Start time: {datetime.now().isoformat()}")
        logger.info("="*50)
        
        app = TextToSQLApp()
        app.run()
        
        logger.info("Application session completed successfully")
        
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}", exc_info=True)
        st.error("Critical application error. Please check the logs.")
        raise
    
    finally:
        logger.info("Application shutdown")