import logging
from typing import Dict, List, Any, Optional
import json
import os
import pandas as pd
import numpy as np

# Import configuration
from config import settings, AGENT_CONFIGS, get_llm

# Configure logging
logger = logging.getLogger(__name__)

class AnalysisAgent:
    """
    Analysis Agent is responsible for performing data analysis using Python.
    It processes data and provides insights through statistical analysis.
    """
    
    def __init__(self):
        """Initialize the Analysis Agent"""
        # Create the LLM using the helper function
        self.llm = get_llm("analysis_agent")
        
        # Create the analysis planning prompt
        self.analysis_prompt = """
You are a data analyst for a university administration system.
You need to analyze the following data and extract meaningful insights.

Data: {data_sample}

Column names: {column_names}

Analysis task: {task}

Please provide a detailed analysis of this data, including key statistics, patterns, and insights.
Format your response as a clear summary with bullet points for key findings.
"""
    
    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data based on the provided task
        
        Args:
            input_data: Dictionary containing task and data information
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Extract information from input
            task = input_data.get("task", "")
            data = input_data.get("data", [])
            column_names = input_data.get("column_names", [])
            
            # Create a DataFrame for analysis
            df = pd.DataFrame(data)
            
            # Perform basic analysis
            analysis_results = self._analyze_dataframe(df, task)
            
            # Get insights from LLM
            # Prepare data sample for prompt (limit to 5 rows for brevity)
            data_sample = str(data[:5] if len(data) > 5 else data)
            
            # Format the prompt
            formatted_prompt = self.analysis_prompt.format(
                task=task,
                column_names=column_names,
                data_sample=data_sample
            )
            
            # Get analysis insights
            response = self.llm.invoke(formatted_prompt)
            summary = response.content
            
            # Combine everything
            return {
                "summary": summary,
                "details": analysis_results,
                "data_sample": data[:3] if data else []
            }
            
        except Exception as e:
            logger.error(f"Error in Analysis Agent: {e}", exc_info=True)
            return self._generate_basic_analysis(task, data, column_names)
    
    def _analyze_dataframe(self, df: pd.DataFrame, task: str) -> Dict[str, Any]:
        """
        Perform basic analysis on a DataFrame
        
        Args:
            df: DataFrame to analyze
            task: Analysis task description
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # Basic info
        analysis["row_count"] = len(df)
        analysis["column_count"] = len(df.columns)
        
        # Column statistics
        column_stats = {}
        for col in df.columns:
            # Skip columns with all null values
            if df[col].isna().all():
                continue
                
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Numeric column
                    column_stats[col] = {
                        "type": "numeric",
                        "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                        "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                        "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                        "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                        "null_count": int(df[col].isna().sum())
                    }
                else:
                    # Categorical/text column
                    value_counts = df[col].value_counts().head(5).to_dict()
                    column_stats[col] = {
                        "type": "categorical",
                        "unique_values": int(df[col].nunique()),
                        "top_values": {str(k): int(v) for k, v in value_counts.items()},
                        "null_count": int(df[col].isna().sum())
                    }
            except Exception as e:
                # If analysis fails for this column
                column_stats[col] = {
                    "type": "unknown",
                    "error": str(e)
                }
        
        analysis["column_stats"] = column_stats
        
        # Task-specific analysis
        if "count" in task.lower() or "how many" in task.lower():
            analysis["count_result"] = len(df)
            
        return analysis
    
    def _generate_basic_analysis(self, task: str, data: List[Dict[str, Any]], column_names: List[str]) -> Dict[str, Any]:
        """
        Generate basic analysis as a fallback
        
        Args:
            task: The original analysis task
            data: The data to analyze
            column_names: Column names in the data
            
        Returns:
            Dictionary with basic analysis
        """
        try:
            # Generate a basic summary
            summary = f"Analyzed {len(data)} records with {len(column_names)} attributes."
            
            if "count" in task.lower() or "how many" in task.lower():
                summary = f"There are {len(data)} records in the dataset."
            
            return {
                "summary": summary,
                "details": {
                    "record_count": len(data),
                    "column_count": len(column_names),
                    "columns": column_names
                },
                "is_fallback": True
            }
        except Exception as e:
            logger.error(f"Error in basic analysis fallback: {e}")
            return {
                "summary": f"Unable to analyze the data due to an error: {str(e)}",
                "details": {},
                "is_fallback": True
            }