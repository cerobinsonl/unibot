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
You are the Data Analysis Agent for a university administrative system.
Your expertise is analyzing data using Python to find patterns, trends, and insights.

You need to create a Python code snippet that analyzes data based on a specific task.

The code should:
1. Use pandas and numpy for data manipulation and analysis
2. Include descriptive statistics, aggregations, or trend analysis as appropriate
3. Identify key patterns, outliers, or insights
4. Create a concise summary of findings
5. Return both the summary and detailed results

Format your response as a JSON object with these keys:
- code: The Python code that will perform the analysis
- explanation: Brief explanation of the analytical approach

Your code will receive a pandas DataFrame called 'df' with column names as provided.
Your code should return a dictionary with:
- 'summary': A text summary of key findings (1-3 paragraphs)
- 'details': A dictionary of detailed analysis results

Analysis task: {task}

Column names: {column_names}

Data sample: {data_sample}

Please generate the analysis code based on this information.
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
            
            # Prepare data sample for prompt (limit to 5 rows for brevity)
            data_sample = str(data[:5])
            
            # Get analysis plan
            formatted_prompt = self.analysis_prompt.format(
                task=task,
                column_names=column_names,
                data_sample=data_sample
            )
            
            analysis_response = self.llm.invoke(formatted_prompt).content
            
            # Parse the code from the response
            try:
                response_json = json.loads(analysis_response)
                code = response_json.get("code", "")
                explanation = response_json.get("explanation", "")
            except json.JSONDecodeError:
                # If not valid JSON, try to extract code using regex
                import re
                code_match = re.search(r'```python\s*(.*?)\s*```', analysis_response, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                else:
                    # Last attempt to find Python code
                    code_match = re.search(r'import pandas|import numpy(.*?)(?:```|$)', analysis_response, re.DOTALL)
                    code = code_match.group(0) if code_match else ""
                
                explanation = "Analysis code extracted from non-JSON response"
            
            if not code:
                raise ValueError("No analysis code could be extracted from the response")
            
            # Execute the analysis code
            analysis_result = self._execute_analysis(code, data)
            
            # Add explanation to the result
            analysis_result["explanation"] = explanation
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in Analysis Agent: {e}", exc_info=True)
            
            # For the POC, generate basic analysis on error
            if os.getenv("BASIC_ANALYSIS_ON_ERROR", "true").lower() == "true":
                return self._generate_basic_analysis(task, data, column_names)
            
            raise e
    
    def _execute_analysis(self, code: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute the analysis code safely
        
        Args:
            code: Python code to execute
            data: Data to analyze
            
        Returns:
            Analysis results
        """
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            
            # Create a safe execution environment
            local_vars = {"df": df, "pd": pd, "np": np}
            
            # Execute the code
            exec(code, {"pd": pd, "np": np}, local_vars)
            
            # Get the results (should be a dictionary with 'summary' and 'details')
            if "results" in local_vars:
                results = local_vars["results"]
                
                # Ensure proper format
                if not isinstance(results, dict):
                    results = {"summary": str(results), "details": {}}
                
                if "summary" not in results:
                    results["summary"] = "Analysis completed but no summary provided."
                
                if "details" not in results:
                    results["details"] = {}
                
                return results
            else:
                # No results variable found, create default
                return {
                    "summary": "Analysis completed but no results were returned.",
                    "details": {}
                }
                
        except Exception as e:
            logger.error(f"Error executing analysis code: {e}", exc_info=True)
            raise ValueError(f"Error executing analysis code: {str(e)}")
    
    def _generate_basic_analysis(self, task: str, data: List[Dict[str, Any]], column_names: List[str]) -> Dict[str, Any]:
        """
        Generate basic analysis when the main analysis fails
        
        Args:
            task: The original analysis task
            data: The data to analyze
            column_names: Column names in the data
            
        Returns:
            Dictionary with basic analysis
        """
        logger.info("Generating basic analysis fallback")
        
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            
            # Initialize results
            summary = []
            details = {}
            
            # Basic row and column count
            row_count = len(df)
            col_count = len(df.columns)
            summary.append(f"The dataset contains {row_count} records with {col_count} attributes.")
            
            # Add column information
            details["column_info"] = {}
            
            for col in df.columns:
                col_type = str(df[col].dtype)
                details["column_info"][col] = {"type": col_type}
                
                # For numeric columns, add basic stats
                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        stats = {
                            "min": float(df[col].min()),
                            "max": float(df[col].max()),
                            "mean": float(df[col].mean()),
                            "median": float(df[col].median()),
                            "std": float(df[col].std())
                        }
                        details["column_info"][col]["stats"] = stats
                        
                        # Add to summary for key metrics
                        summary.append(f"The {col} ranges from {stats['min']:.2f} to {stats['max']:.2f} with an average of {stats['mean']:.2f}.")
                    except:
                        pass
                
                # For categorical columns, show value counts
                elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    try:
                        value_counts = df[col].value_counts().to_dict()
                        # Limit to top 10 values
                        if len(value_counts) > 10:
                            top_values = dict(sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:10])
                            details["column_info"][col]["top_values"] = top_values
                            
                            # Add to summary
                            top_category = max(value_counts.items(), key=lambda x: x[1])[0]
                            summary.append(f"The most common value in {col} is '{top_category}' which appears {value_counts[top_category]} times.")
                        else:
                            details["column_info"][col]["values"] = value_counts
                    except:
                        pass
            
            # Look for missing values
            missing_values = df.isnull().sum().to_dict()
            any_missing = any(count > 0 for count in missing_values.values())
            if any_missing:
                details["missing_values"] = {col: count for col, count in missing_values.items() if count > 0}
                summary.append(f"Some columns contain missing values: {', '.join(details['missing_values'].keys())}.")
            
            # Create correlations for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 2:
                try:
                    corr = df[numeric_cols].corr().round(2).to_dict()
                    details["correlations"] = corr
                    
                    # Find strongest correlation
                    strongest = 0
                    col1, col2 = None, None
                    for c1 in corr:
                        for c2 in corr[c1]:
                            if c1 != c2 and abs(corr[c1][c2]) > strongest:
                                strongest = abs(corr[c1][c2])
                                col1, col2 = c1, c2
                    
                    if col1 and col2:
                        corr_direction = "positive" if corr[col1][col2] > 0 else "negative"
                        summary.append(f"There is a strong {corr_direction} correlation ({corr[col1][col2]:.2f}) between {col1} and {col2}.")
                except:
                    pass
            
            # Create final summary text
            summary_text = " ".join(summary)
            
            return {
                "summary": summary_text,
                "details": details,
                "is_fallback": True
            }
            
        except Exception as e:
            logger.error(f"Error in basic analysis fallback: {e}", exc_info=True)
            
            # If even the basic analysis fails, return minimal information
            return {
                "summary": f"Unable to analyze the data due to an error: {str(e)}",
                "details": {},
                "is_fallback": True
            }