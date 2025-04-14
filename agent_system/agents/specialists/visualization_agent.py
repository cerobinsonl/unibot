import logging
from typing import Dict, List, Any, Optional
import json
import base64
import io
import os
import re

# Import visualization tools
from tools.visualization import create_visualization

# Import configuration
from config import settings, AGENT_CONFIGS, get_llm

# Configure logging
logger = logging.getLogger(__name__)

class VisualizationAgent:
    """
    Visualization Agent is responsible for creating visual representations
    of data using Python visualization libraries.
    """
    
    def __init__(self):
        """Initialize the Visualization Agent"""
        # Create the LLM using the helper function
        self.llm = get_llm("visualization_agent")
        
        # Create the visualization planning prompt
        self.visualization_prompt = """
You are the Visualization Agent for a university administrative system.
Your specialty is creating clear, insightful visualizations using Python libraries.

You need to create a Python code snippet that generates a visualization based on provided data.

The code should:
1. Use matplotlib, seaborn, or plotly
2. Create a clear, informative visualization appropriate for the data
3. Include proper titles, labels, and legends
4. Use a professional color scheme suitable for university reporting
5. Handle any data transformation needed for visualization
6. Save the plot to a BytesIO object for display

Format your response as a JSON object with these keys:
- chart_type: The type of chart you're creating (e.g., "bar", "line", "scatter", "pie")
- code: The Python code that will generate the visualization
- explanation: Brief explanation of why this visualization is appropriate

Your code will receive a pandas DataFrame called 'df' with column names as provided.

Visualization task: {task}

Column names: {column_names}

Data sample: {data_sample}

Analysis summary: {analysis_summary}

Please generate the visualization code based on this information.
"""
    
    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a visualization based on the provided data
        
        Args:
            input_data: Dictionary containing task and data information
            
        Returns:
            Dictionary containing visualization data
        """
        try:
            # Extract information from input
            task = input_data.get("task", "")
            data = input_data.get("data", [])
            column_names = input_data.get("column_names", [])
            analysis = input_data.get("analysis", {})
            
            # Prepare data sample for prompt (limit to 5 rows for brevity)
            data_sample = str(data[:5])
            
            # Get analysis summary if available
            analysis_summary = analysis.get("summary", "No analysis summary provided")
            
            # Format the prompt with the required values
            formatted_prompt = self.visualization_prompt.format(
                task=task,
                column_names=column_names,
                data_sample=data_sample,
                analysis_summary=analysis_summary
            )
            
            # Get visualization plan
            visualization_response = self.llm.invoke(formatted_prompt)
            
            # Extract generated content
            content = visualization_response.content
            
            # Parse the code from the response
            try:
                response_json = json.loads(content)
                code = response_json.get("code", "")
                chart_type = response_json.get("chart_type", "unknown")
                explanation = response_json.get("explanation", "")
            except json.JSONDecodeError:
                # If not valid JSON, try to extract code using regex
                code_match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                else:
                    # Last attempt to find Python code
                    code_match = re.search(r'import matplotlib|import seaborn|import plotly(.*?)(?:```|$)', content, re.DOTALL)
                    code = code_match.group(0) if code_match else ""
                
                chart_type = "unknown"
                explanation = "Visualization code extracted from non-JSON response"
            
            if not code:
                raise ValueError("No visualization code could be extracted from the response")
            
            # Create the visualization using the extracted code
            image_data, image_format = create_visualization(code, data)
            
            # Encode image as base64 for transmission
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Return the visualization
            return {
                "image_data": base64_image,
                "image_type": f"image/{image_format}",
                "chart_type": chart_type,
                "explanation": explanation
            }
            
        except Exception as e:
            logger.error(f"Error in Visualization Agent: {e}", exc_info=True)
            
            # For the POC, return a fallback visualization on error
            if os.getenv("MOCK_VISUALIZATION_ON_ERROR", "true").lower() == "true":
                return self._generate_fallback_visualization(task, data, column_names)
            
            raise e
    
    def _generate_fallback_visualization(self, task: str, data: List[Dict[str, Any]], column_names: List[str]) -> Dict[str, Any]:
        """
        Generate a simple fallback visualization when the main one fails
        
        Args:
            task: The original visualization task
            data: The data to visualize
            column_names: Column names in the data
            
        Returns:
            Dictionary with visualization data
        """
        logger.info("Generating fallback visualization")
        
        try:
            # Generate a very simple bar chart as fallback
            import matplotlib.pyplot as plt
            import pandas as pd
            import io
            
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            
            if len(df) == 0 or len(df.columns) == 0:
                # If no data, create dummy data
                df = pd.DataFrame({
                    'Category': ['A', 'B', 'C', 'D', 'E'],
                    'Value': [5, 7, 3, 9, 4]
                })
            
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) > 0:
                # Use the first numeric column
                value_col = numeric_cols[0]
                
                # Get a category column if possible
                category_cols = df.select_dtypes(include=['object']).columns
                if len(category_cols) > 0:
                    category_col = category_cols[0]
                    # Limit to top 10 categories for readability
                    top_data = df.groupby(category_col)[value_col].sum().nlargest(10).reset_index()
                    
                    plt.figure(figsize=(10, 6))
                    plt.bar(top_data[category_col], top_data[value_col])
                    plt.title(f"Fallback Visualization: {category_col} vs {value_col}")
                    plt.xlabel(category_col)
                    plt.ylabel(value_col)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                else:
                    # No category column, just plot the numeric column
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(len(df)), df[value_col])
                    plt.title(f"Fallback Visualization: {value_col}")
                    plt.xlabel("Index")
                    plt.ylabel(value_col)
                    plt.tight_layout()
            else:
                # No numeric columns, create a count plot of a category
                if len(df.columns) > 0:
                    col = df.columns[0]
                    counts = df[col].value_counts().nlargest(10)
                    
                    plt.figure(figsize=(10, 6))
                    plt.bar(counts.index, counts.values)
                    plt.title(f"Fallback Visualization: Counts of {col}")
                    plt.xlabel(col)
                    plt.ylabel("Count")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                else:
                    # Last resort - empty dataframe
                    plt.figure(figsize=(10, 6))
                    plt.text(0.5, 0.5, "No data available for visualization", 
                             horizontalalignment='center', verticalalignment='center',
                             transform=plt.gca().transAxes)
                    plt.axis('off')
            
            # Save the figure to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=settings.VISUALIZATION_DPI)
            buf.seek(0)
            
            # Encode as base64
            base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            return {
                "image_data": base64_image,
                "image_type": "image/png",
                "chart_type": "fallback_bar_chart",
                "explanation": "This is a fallback visualization created when the main visualization generation encountered an error."
            }
            
        except Exception as e:
            logger.error(f"Error in fallback visualization: {e}", exc_info=True)
            
            # If even the fallback fails, return a message
            return {
                "image_data": None,
                "image_type": None,
                "chart_type": None,
                "explanation": "Visualization could not be generated due to an error."
            }