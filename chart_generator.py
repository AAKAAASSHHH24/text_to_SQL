
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from loguru import logger
from decorators import log_execution, log_errors
import time

class ChartGenerator:
    """Enhanced chart generator with comprehensive logging"""
    
    @staticmethod
    @log_execution(include_args=True)
    def analyze_data_for_charting(data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data structure for chart generation with logging"""
        logger.info(f"Analyzing data for charting - Shape: {data.shape}")
        
        analysis = {
            "numeric_columns": data.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": data.select_dtypes(include=['object']).columns.tolist(),
            "datetime_columns": data.select_dtypes(include=['datetime64']).columns.tolist(),
            "row_count": len(data),
            "column_count": len(data.columns),
            "has_nulls": data.isnull().any().any(),
            "memory_usage": data.memory_usage(deep=True).sum()
        }
        
        logger.debug(f"Data analysis result: {analysis}")
        return analysis
    
    @staticmethod
    @log_execution(include_args=True)
    def determine_chart_type(query: str, data: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Determine appropriate chart type with enhanced logging"""
        logger.info(f"Determining chart type for query: '{query[:50]}...'")
        
        # Analyze data first
        data_analysis = ChartGenerator.analyze_data_for_charting(data)
        
        if data.empty:
            logger.warning("Cannot create chart: Data is empty")
            return None, None, None
        
        if data_analysis["column_count"] < 2:
            logger.warning(f"Cannot create chart: Need at least 2 columns, got {data_analysis['column_count']}")
            return None, None, None
        
        if not data_analysis["numeric_columns"]:
            logger.warning("Cannot create chart: No numeric columns found")
            return None, None, None
        
        # Query context analysis
        query_lower = query.lower()
        
        context_analysis = {
            "temporal_keywords": ['time', 'date', 'year', 'month', 'day', 'trend', 'over time'],
            "comparison_keywords": ['compare', 'vs', 'versus', 'top', 'bottom', 'most', 'least', 'ranking'],
            "distribution_keywords": ['distribution', 'percentage', 'proportion', 'share', 'breakdown']
        }
        
        context_scores = {}
        for context_type, keywords in context_analysis.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            context_scores[context_type] = score
        
        logger.debug(f"Context analysis scores: {context_scores}")
        
        # Determine best columns for charting
        x_col = (data_analysis["categorical_columns"][0] if data_analysis["categorical_columns"] 
                else data.columns[0])
        y_col = data_analysis["numeric_columns"][0]
        
        logger.debug(f"Selected columns - X: {x_col}, Y: {y_col}")
        
        # Chart type decision logic
        chart_type = None
        decision_reason = ""
        
        if (context_scores["temporal_keywords"] > 0 or 
            any('date' in col.lower() for col in data.columns) or
            data_analysis["datetime_columns"]):
            chart_type = 'line'
            decision_reason = "Temporal context detected"
            
        elif (context_scores["distribution_keywords"] > 0 and 
              data_analysis["row_count"] <= 10):
            chart_type = 'pie'
            decision_reason = "Distribution context with small dataset"
            
        elif (context_scores["comparison_keywords"] > 0 or 
              data_analysis["categorical_columns"]):
            chart_type = 'bar'
            decision_reason = "Comparison context or categorical data"
            
        elif data_analysis["row_count"] <= 20:
            chart_type = 'bar'
            decision_reason = "Small dataset size"
            
        else:
            chart_type = 'line'
            decision_reason = "Default for larger dataset"
        
        logger.info(f"Chart decision: {chart_type} - Reason: {decision_reason}")
        return chart_type, x_col, y_col
    
    @staticmethod
    @log_execution(include_args=True)
    @log_errors("Chart creation failed")
    def create_chart(chart_type: str, data: pd.DataFrame, x_col: str, y_col: str):
        """Create chart with comprehensive logging and error handling"""
        logger.info(f"Creating {chart_type} chart - X: {x_col}, Y: {y_col}")
        
        start_time = time.time()
        
        try:
            # Pre-creation validation
            if x_col not in data.columns:
                raise ValueError(f"X column '{x_col}' not found in data")
            if y_col not in data.columns:
                raise ValueError(f"Y column '{y_col}' not found in data")
            
            # Data preparation logging
            logger.debug(f"X column data type: {data[x_col].dtype}")
            logger.debug(f"Y column data type: {data[y_col].dtype}")
            logger.debug(f"X column unique values: {data[x_col].nunique()}")
            logger.debug(f"Y column range: {data[y_col].min()} to {data[y_col].max()}")
            
            # Create chart based on type
            fig = None
            
            if chart_type == 'bar':
                logger.debug("Creating bar chart with plotly express")
                fig = px.bar(
                    data, 
                    x=x_col, 
                    y=y_col, 
                    title=f"{y_col} by {x_col}",
                    labels={x_col: x_col.replace('_', ' ').title(),
                           y_col: y_col.replace('_', ' ').title()}
                )
                
            elif chart_type == 'pie':
                logger.debug("Creating pie chart with plotly express")
                fig = px.pie(
                    data, 
                    names=x_col, 
                    values=y_col, 
                    title=f"{y_col} Distribution by {x_col}"
                )
                
            elif chart_type == 'line':
                logger.debug("Creating line chart with plotly express")
                fig = px.line(
                    data, 
                    x=x_col, 
                    y=y_col, 
                    title=f"{y_col} Trend by {x_col}",
                    labels={x_col: x_col.replace('_', ' ').title(),
                           y_col: y_col.replace('_', ' ').title()}
                )
                fig.update_traces(mode='lines+markers')
                
            else:
                logger.error(f"Unknown chart type: {chart_type}")
                raise ValueError(f"Unsupported chart type: {chart_type}")
            
            # Enhance chart appearance
            fig.update_layout(
                height=400,
                font=dict(size=12),
                title_font_size=16,
                showlegend=True if chart_type == 'pie' else False
            )
            
            creation_time = time.time() - start_time
            
            logger.success(f"{chart_type.title()} chart created successfully in {creation_time:.3f}s")
            
            # Log chart metadata
            chart_metadata = {
                "chart_type": chart_type,
                "data_points": len(data),
                "x_column": x_col,
                "y_column": y_col,
                "creation_time": creation_time
            }
            
            logger.debug(f"Chart metadata: {chart_metadata}")
            
            return fig
            
        except Exception as e:
            creation_time = time.time() - start_time
            logger.error(f"Chart creation failed after {creation_time:.3f}s: {str(e)}", exc_info=True)
            raise Exception(f"Chart creation failed: {str(e)}")