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
You need to analyze the provided data and extract meaningful insights.

DATA INFORMATION:
Column names: {column_names}
Number of records: {row_count}
Data sample: {data_sample}

ANALYSIS TASK:
{task}

DATA INSIGHTS:
{data_insights}

IMPORTANT GUIDELINES:
1. Analyze ONLY the data structure provided
2. Do not make assumptions about columns that don't exist in the data
3. Focus your analysis on columns that are relevant to the task
4. For numerical columns, calculate appropriate statistics (mean, median, range, distribution)
5. For categorical columns, analyze frequencies and distributions
6. Identify any patterns, outliers, or interesting insights in the data
7. For time-series data, look for trends or seasonal patterns
8. Note any data quality issues (missing values, unusual distributions)
9. Consider the context of university administration in your analysis
10. Provide specific, actionable insights whenever possible

Your analysis should include:
- Summary of key findings
- Detailed statistical analysis of relevant columns
- Distribution information for key metrics
- Any patterns or relationships between variables
- Recommendations based on the data (if appropriate)

Please provide a comprehensive analysis focused specifically on the provided data structure and the requested task.
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
            row_count = input_data.get("row_count", len(data) if data else 0)
            
            # Log input data information
            logger.info(f"Analyzing data with {row_count} rows and {len(column_names)} columns")
            logger.info(f"Column names: {column_names}")
            if data and len(data) > 0:
                logger.info(f"Sample data (first record): {data[0]}")
            
            # Validate data before analysis
            if not data or len(data) == 0:
                logger.warning("No data to analyze")
                return {
                    "summary": "No data available for analysis.",
                    "details": {"error": "No data provided"},
                    "data_sample": []
                }
            
            # Create a DataFrame for analysis
            df = pd.DataFrame(data)
            
            # Check for empty dataframe even with data input
            if df.empty:
                logger.warning("DataFrame is empty after conversion")
                return {
                    "summary": "The provided data could not be processed for analysis.",
                    "details": {"error": "Empty DataFrame after conversion"},
                    "data_sample": data[:3] if data else []
                }
            
            # Check for column existence if column_names were provided
            for col in column_names:
                if col not in df.columns:
                    logger.warning(f"Column '{col}' mentioned in column_names not found in data")
            
            # Log column types and null value counts
            col_types = {col: str(df[col].dtype) for col in df.columns}
            null_counts = {col: int(df[col].isna().sum()) for col in df.columns}
            logger.info(f"Column types: {col_types}")
            logger.info(f"Null value counts: {null_counts}")
            
            # Get detailed dataframe info
            df_info = self._get_dataframe_info(df)
            logger.info(f"Obtained dataframe info with {len(df_info)} keys")
            
            # Perform basic analysis
            analysis_results = self._analyze_dataframe(df, task)
            
            # Calculate distributions for relevant numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if len(df[col].dropna()) > 5:  # Only if we have enough non-null values
                    try:
                        analysis_results[f"{col}_distribution"] = self._calculate_distribution(df[col], bins=10)
                    except Exception as dist_err:
                        logger.warning(f"Error calculating distribution for {col}: {dist_err}")
            
            # Convert analysis_results to a JSON-serializable format (handle numpy types)
            analysis_results = self._convert_to_serializable(analysis_results)
            
            # Get insights from LLM
            # Prepare data sample for prompt (limit to 5 rows for brevity)
            data_sample = str(data[:5] if len(data) > 5 else data)
            
            # Format the prompt with enhanced data insights
            enhanced_prompt = self.analysis_prompt.format(
                task=task,
                column_names=column_names,
                row_count=row_count,
                data_sample=data_sample,
                data_insights=json.dumps(analysis_results, indent=2)
            )
            
            # Get analysis insights
            response = self.llm.invoke(enhanced_prompt)
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
                column_stats[col] = {"type": "unknown", "null_count": len(df), "all_null": True}
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
                        "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                        "null_count": int(df[col].isna().sum()),
                        "unique_count": int(df[col].nunique())
                    }
                else:
                    # Categorical/text column
                    value_counts = df[col].value_counts().head(10).to_dict()
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
            
        # If GPA is mentioned in the task, look for GPA columns
        if "gpa" in task.lower():
            gpa_cols = [col for col in df.columns if "gpa" in col.lower()]
            if gpa_cols:
                analysis["gpa_analysis"] = {}
                for gpa_col in gpa_cols:
                    if pd.api.types.is_numeric_dtype(df[gpa_col]):
                        gpa_data = df[gpa_col].dropna()
                        analysis["gpa_analysis"][gpa_col] = {
                            "mean": float(gpa_data.mean()) if not pd.isna(gpa_data.mean()) else None,
                            "median": float(gpa_data.median()) if not pd.isna(gpa_data.median()) else None,
                            "std": float(gpa_data.std()) if not pd.isna(gpa_data.std()) else None,
                            "min": float(gpa_data.min()) if not pd.isna(gpa_data.min()) else None,
                            "max": float(gpa_data.max()) if not pd.isna(gpa_data.max()) else None,
                            "count": len(gpa_data)
                        }
        
        # If looking for top values or rankings
        if any(term in task.lower() for term in ["top", "highest", "most", "best", "ranking"]):
            for col in df.select_dtypes(include=['number']).columns:
                if len(df[col].dropna()) > 0:
                    try:
                        # Get top 5 values
                        top_values = df.nlargest(5, col)
                        analysis[f"top_{col}"] = top_values.to_dict(orient="records")
                    except:
                        pass
        
        return analysis
    
    def _get_dataframe_info(self, df):
        """
        Get detailed information about the dataframe structure
        
        Args:
            df: Pandas DataFrame to analyze
            
        Returns:
            Dictionary with dataframe information
        """
        info = {}
        
        # Basic info
        info["row_count"] = len(df)
        info["column_count"] = len(df.columns)
        info["column_names"] = list(df.columns)
        
        # Column types
        info["column_types"] = {col: str(df[col].dtype) for col in df.columns}
        
        # Null counts
        info["null_counts"] = {col: int(df[col].isna().sum()) for col in df.columns}
        
        # Column statistics
        column_stats = {}
        
        # For numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            try:
                column_stats[col] = {
                    "type": "numeric",
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                    "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                    "unique_count": int(df[col].nunique()),
                    "non_null_count": int(df[col].count())
                }
            except:
                # Skip if stats calculation fails
                pass
        
        # For categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            try:
                value_counts = df[col].value_counts().head(5).to_dict()
                column_stats[col] = {
                    "type": "categorical",
                    "unique_count": int(df[col].nunique()),
                    "non_null_count": int(df[col].count()),
                    "top_values": {str(k): int(v) for k, v in value_counts.items()}
                }
            except:
                # Skip if stats calculation fails
                pass
        
        # For datetime columns
        date_cols = df.select_dtypes(include=['datetime']).columns
        for col in date_cols:
            try:
                column_stats[col] = {
                    "type": "datetime",
                    "min": df[col].min().strftime("%Y-%m-%d %H:%M:%S") if not pd.isna(df[col].min()) else None,
                    "max": df[col].max().strftime("%Y-%m-%d %H:%M:%S") if not pd.isna(df[col].max()) else None,
                    "unique_count": int(df[col].nunique()),
                    "non_null_count": int(df[col].count())
                }
            except:
                # Skip if stats calculation fails
                pass
        
        info["column_stats"] = column_stats
        
        return info
    
    def _calculate_distribution(self, series, bins=10):
        """
        Calculate the distribution of values in a series
        
        Args:
            series: Pandas Series to analyze
            bins: Number of bins for histogram
            
        Returns:
            Dictionary with bin information
        """
        try:
            # Drop NA values
            series = series.dropna()
            
            # Create histogram
            hist, bin_edges = np.histogram(series, bins=bins)
            
            # Convert to list for JSON serialization
            hist_list = hist.tolist()
            bin_edges_list = bin_edges.tolist()
            
            # Create bins with ranges and counts
            bins_info = []
            for i in range(len(hist_list)):
                bin_info = {
                    "range": f"{bin_edges_list[i]:.2f} - {bin_edges_list[i+1]:.2f}",
                    "count": int(hist_list[i])
                }
                bins_info.append(bin_info)
            
            return bins_info
        except Exception as e:
            logger.error(f"Error calculating distribution: {e}")
            return []
    
    def _convert_to_serializable(self, obj):
        """
        Convert object to JSON-serializable format
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_serializable(obj.tolist())
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            return str(obj)
    
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