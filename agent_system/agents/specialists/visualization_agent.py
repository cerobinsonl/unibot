import logging
from typing import Dict, List, Any, Optional, Tuple
import json
import base64
import io
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

You need to create a Python code snippet that defines a function to generate a visualization based on provided data,
and then immediately calls that function at the end of the snippet.

IMPORTANT DATA INFORMATION:
{dataframe_info}

TASK:
{task}

Column names available: {column_names}

Analysis summary: {analysis_summary}

The code should:
1. Define a function (e.g., `create_visualization(data, buffer)`) that:
   - Accepts a pandas DataFrame (`data`) and a BytesIO object (`buffer`).
   - Performs any necessary data validation and transformation.
   - Uses matplotlib, seaborn, or plotly to build the appropriate chart.
   - Sets titles, labels, legends, and a professional color scheme.
   - Saves the figure to the provided `buffer` using `plt.savefig(buffer, format='png', dpi=100)`, then `buffer.seek(0)`.
2. After the function definition, call that function once using the variables `data` (the DataFrame) and `buffer` (the BytesIO).
3. **DO NOT** generate something like this: 
    # Example usage (assuming 'data' DataFrame and 'buffer' BytesIO are defined)
    import pandas as pd
    import io
    
    # Create a sample DataFrame (replace with your actual data)
    data = pd.DataFrame({'GPA': [3.2, 3.5, 3.8, 2.9, 3.1, 3.3, 3.6, 4.0, 2.5, 3.9] * 100})
    
    # Create a BytesIO buffer
    buffer = io.BytesIO()

GUIDELINES:
- Use only the actual column names from `{column_names}`.
- Include error handling for missing or invalid data.
- Ensure the function is self-contained and executable.

FORMAT YOUR RESPONSE AS JSON with keys:
- `chart_type`: The type of chart ("bar", "histogram", etc.)
- `code`: The full Python snippet, including the function definition and its invocation.
- `explanation`: Brief rationale for the chosen visualization and function structure.

Do not include any usage beyond the single function call at the end.
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
            
            logger.info(f"Creating visualization with task: {task[:100]}...")
            logger.info(f"Data has {len(data)} records and {len(column_names)} columns")
            
            # Check if data is empty or insufficient
            if not data or len(data) < 1:
                logger.warning(f"Insufficient data for visualization: {len(data) if data else 0} records")
                return self._generate_no_data_visualization("No data available for visualization.")
            
            # Convert to DataFrame for easier inspection
            df = pd.DataFrame(data)
            
            # Log data info
            logger.info(f"DataFrame info: {len(df)} rows × {len(df.columns)} columns")
            logger.info(f"DataFrame columns: {list(df.columns)}")
            
            # Prepare dataframe information for the LLM
            dataframe_info = self._prepare_dataframe_info(df)
            
            # Prepare data sample for prompt (limit to 5 rows for brevity)
            data_sample = str(data[:5] if len(data) > 5 else data)
            
            # Get analysis summary if available
            analysis_summary = analysis.get("summary", "No analysis summary provided")
            
            # Format the prompt with the required values
            formatted_prompt = self.visualization_prompt.format(
                task=task,
                column_names=column_names,
                data_sample=data_sample,
                analysis_summary=analysis_summary,
                dataframe_info=dataframe_info
            )
            
            # Get visualization plan
            with open("/app/visual_debug/debug_data.json", "w") as f:
                json.dump(data[:10], f, indent=2)

            # Log the count too
            logger.info(f"VisualizationAgent debug: received {len(data)} data rows. Wrote debug_data.json.")
            visualization_response = self.llm.invoke(formatted_prompt)
            
            # Extract generated content
            content = visualization_response.content
            if content.strip().startswith("```"):
                # remove leading ```json (or ```) and trailing ```
                content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
                content = re.sub(r"\s*```$", "", content, flags=re.MULTILINE)
            
            content = re.sub(r"\\\s*\n", "", content)  
            with open("/app/visual_debug/debug_llm_output.txt", "w") as f:
                f.write(content)

            logger.info("VisualizationAgent debug: wrote raw LLM output to debug_llm_output.txt")


            # Parse the code from the response
            try:
                response_json = json.loads(content)


                code = response_json["code"]
                code = re.sub(r'# Sample data[\\s\\S]+$', '', code)
                
                with open("/app/visual_debug/debug_extracted_code.py", "w") as f:
                    f.write(code)

                logger.info("VisualizationAgent debug 1: wrote extracted code to debug_extracted_code.py")

                chart_type = response_json.get("chart_type", "unknown")
                explanation = response_json.get("explanation", "")
            except json.JSONDecodeError:
                logger.warning("Failed to parse visualization response as JSON, attempting regex extraction")
                # If not valid JSON, try to extract code using regex
                code_match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                    logger.info("Successfully extracted code using ```python``` pattern")
                else:
                    # Last attempt to find Python code
                    code_match = re.search(r'import matplotlib|import seaborn|import plotly(.*?)(?:```|$)', content, re.DOTALL)
                    code = code_match.group(0) if code_match else ""
                    # Remove any “# Sample data” and everything that follows it
                    code = re.sub(r'# Sample data[\\s\\S]+$', '', code)
                    with open("/app/visual_debug/debug_extracted_code.py", "w") as f:
                        f.write(code)

                    logger.info("VisualizationAgent debug 2: wrote extracted code to debug_extracted_code.py")
                    logger.info("Attempted extraction using import pattern")
                
                # Try to guess chart type from code
                chart_type = "unknown"
                if "hist" in code.lower():
                    chart_type = "histogram"
                elif "pie" in code.lower():
                    chart_type = "pie"
                elif "bar" in code.lower():
                    chart_type = "bar"
                elif "line" in code.lower():
                    chart_type = "line"
                elif "scatter" in code.lower():
                    chart_type = "scatter"
                
                explanation = "Visualization code extracted from non-JSON response"
            
            if not code:
                logger.warning("No visualization code could be extracted from the response")
                return self._generate_no_data_visualization("Couldn't generate appropriate visualization code.")
            
            # Add proper error handling to the code
            code = self._add_error_handling_to_code(code, df)
            
            # Log the code being used
            logger.debug(f"Visualization code: {code[:500]}...")
            
            # Create the visualization using the extracted code
            logger.info("Executing visualization code...")
            image_data, image_format = self._execute_visualization_code(code, data)
            
            # Check if we have valid image data
            if not image_data or len(image_data) == 0:
                logger.error("No image data returned from _execute_visualization_code")
                return self._generate_error_visualization("Failed to generate visualization: No image data returned")
            
            logger.info(f"Generated image data with size: {len(image_data)} bytes, format: {image_format}")
            
            # Encode image as base64 for transmission
            try:
                base64_image = base64.b64encode(image_data).decode('utf-8')
                logger.info(f"Successfully encoded image to base64, length: {len(base64_image)}")
                
                # Validate the base64 string (to detect corruption)
                try:
                    # Just to validate the base64 is correct
                    test_decode = base64.b64decode(base64_image)
                    logger.info(f"Base64 validation successful, decoded length: {len(test_decode)}")
                except Exception as validate_error:
                    logger.error(f"Base64 validation failed: {validate_error}")
                    # Use a fallback visualization if the encoding is invalid
                    return self._generate_error_visualization(f"Base64 encoding error: {validate_error}")
                
            except Exception as encoding_error:
                logger.error(f"Error encoding image to base64: {encoding_error}")
                return self._generate_error_visualization(f"Image encoding error: {encoding_error}")
            
            # Return the visualization
            result = {
                "image_data": base64_image,
                "image_type": f"image/{image_format}",
                "chart_type": chart_type,
                "explanation": explanation
            }
            
            logger.info(f"Returning visualization result with keys: {list(result.keys())}")
            return result
            
        except Exception as e:
            logger.error(f"Error in Visualization Agent: {e}", exc_info=True)
            return self._generate_error_visualization(str(e))
    
    def _prepare_dataframe_info(self, df):
        """
        Prepare detailed information about the dataframe for the visualization prompt
        
        Args:
            df: Pandas DataFrame to analyze
            
        Returns:
            Formatted string with dataframe information
        """
        try:
            info_parts = []
            
            # Basic info
            info_parts.append(f"DataFrame dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
            info_parts.append(f"Column names: {list(df.columns)}")
            
            # Column types
            col_types = {}
            for col in df.columns:
                col_types[col] = str(df[col].dtype)
            info_parts.append(f"Column types: {col_types}")
            
            # Null counts
            null_counts = {}
            for col in df.columns:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    null_counts[col] = int(null_count)
            
            if null_counts:
                info_parts.append(f"Columns with null values: {null_counts}")
            
            # Column statistics
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                info_parts.append("\nNumeric columns statistics:")
                for col in numeric_cols:
                    try:
                        stats = {
                            "min": float(df[col].min()) if not pd.isna(df[col].min()) else "NULL",
                            "max": float(df[col].max()) if not pd.isna(df[col].max()) else "NULL",
                            "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else "NULL", 
                            "median": float(df[col].median()) if not pd.isna(df[col].median()) else "NULL",
                            "std": float(df[col].std()) if not pd.isna(df[col].std()) else "NULL",
                            "unique_count": int(df[col].nunique())
                        }
                        info_parts.append(f"  - {col}: {stats}")
                    except:
                        pass
            
            # Categorical columns
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                info_parts.append("\nCategorical columns information:")
                for col in cat_cols:
                    try:
                        unique_count = df[col].nunique()
                        info_parts.append(f"  - {col}: {unique_count} unique values")
                        
                        if unique_count < 10:  # Only show values for columns with few unique values
                            value_counts = df[col].value_counts().head(5).to_dict()
                            value_info = ", ".join([f"{k}: {v}" for k, v in value_counts.items()])
                            info_parts.append(f"    Top values: {value_info}")
                    except:
                        pass
            
            # Check for potential date columns
            date_cols = []
            for col in df.columns:
                # Check column name
                if any(date_term in col.lower() for date_term in ['date', 'year', 'month', 'time', 'day']):
                    date_cols.append(col)
                    
            if date_cols:
                info_parts.append(f"\nPotential date/time columns: {date_cols}")
            
            # Check for columns that might be suitable for specific visualizations
            if len(numeric_cols) >= 2:
                info_parts.append("\nPotential scatter plot combinations:")
                for i, col1 in enumerate(numeric_cols[:3]):  # Limit to first 3 to avoid too many combinations
                    for col2 in numeric_cols[i+1:min(i+4, len(numeric_cols))]:
                        info_parts.append(f"  - {col1} vs {col2}")
            
            if len(numeric_cols) >= 1 and len(cat_cols) >= 1:
                info_parts.append("\nPotential bar chart combinations:")
                for cat_col in cat_cols[:2]:  # Limit to first 2 categorical columns
                    for num_col in numeric_cols[:2]:  # Limit to first 2 numeric columns
                        info_parts.append(f"  - {cat_col} (x-axis) vs {num_col} (y-axis)")
            
            # Special cases for university data
            if any('gpa' in col.lower() for col in df.columns):
                info_parts.append("\nGPA data detected - suitable for histogram or density plot")
            
            if any('enrollment' in col.lower() for col in df.columns):
                info_parts.append("\nEnrollment data detected - suitable for trend or bar chart")
            
            if any('aid' in col.lower() for col in df.columns) or any('financial' in col.lower() for col in df.columns):
                info_parts.append("\nFinancial aid data detected - suitable for pie chart or stacked bar chart")
            
            return "\n".join(info_parts)
            
        except Exception as e:
            logger.error(f"Error preparing dataframe info: {e}")
            return f"Error analyzing dataframe: {e}"
    
    def _add_error_handling_to_code(self, code, df):
        """
        Add error handling to visualization code
        
        Args:
            code: The original code
            df: DataFrame to analyze for potential issues
            
        Returns:
            Enhanced code with error handling
        """
        # Check for problematic column names (e.g., with spaces or special characters)
        problem_cols = []
        for col in df.columns:
            if ' ' in col or '.' in col or '-' in col or any(c in col for c in '!"#$%&\'()*+,/:;<=>?@[\\]^`{|}~'):
                problem_cols.append(col)
        
        # Create error handling wrapper
        wrapper_start = """
# Error handling wrapper
try:
    # Convert input to DataFrame if not already
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    
    # Check if DataFrame is empty
    if len(df) == 0:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No data available for visualization", 
                horizontalalignment='center', verticalalignment='center', 
                transform=plt.gca().transAxes, fontsize=14)
        plt.axis('off')
        
        # Save empty chart to buffer
        buf = buffer
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
    else:
"""
        
        # Add column fixes if needed
        if problem_cols:
            wrapper_start += "\n        # Fix problematic column names\n"
            col_fixes = []
            for col in problem_cols:
                safe_col = col.replace(' ', '_').replace('.', '_').replace('-', '_')
                # Remove any remaining special characters
                safe_col = ''.join(c for c in safe_col if c.isalnum() or c == '_')
                col_fixes.append(f'        df = df.rename(columns={{"{col}": "{safe_col}"}})  # Fix problematic column name')
            
            wrapper_start += "\n".join(col_fixes) + "\n"
        
        # Add null value handling
        wrapper_start += """
        # Drop rows with all NaN values
        df = df.dropna(how='all')
        
        # Fill remaining NaN values where needed
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0)
"""
        
        # Indent original code
        indented_code = "\n".join(["        " + line for line in code.split("\n")])
        
        # Wrapper end with error handling
        wrapper_end = """
except Exception as e:
    # Create error visualization
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, f"Error creating visualization: {e}", 
            horizontalalignment='center', verticalalignment='center', 
            transform=plt.gca().transAxes, fontsize=14, color='darkred')
    plt.axis('off')
    
    # Make sure we save to the buffer even if there was an error
    if 'buf' not in locals() and 'buffer' in globals():
        buf = buffer
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
"""
        
        # Combine everything
        enhanced_code = wrapper_start + indented_code + wrapper_end
        
        return enhanced_code
    
    def _execute_visualization_code(self, code: str, data: List[Dict[str, Any]]) -> Tuple[bytes, str]:
        """
        Execute visualization code and get the image data
        
        Args:
            code: Python code to execute
            data: Data to visualize
            
        Returns:
            Tuple of (image data as bytes, image format)
        """
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            logger.info(f"ABER DF {df}")
            # Create a bytes buffer for the image
            buf = io.BytesIO()
            
            # Create a safe execution environment with limited imports
            exec_globals = {
                'pd': pd,
                'plt': plt,
                'sns': sns,
                'np': np,
                'df': df,
                'io': io,
                'buffer': buf
            }
            
            # Add clear figure to avoid contamination from previous runs
            plt.clf()
            
            # Execute the visualization code
            exec(code, exec_globals)
            
            # Check if the code saved the figure to the buffer
            if buf.getbuffer().nbytes == 0:
                # If not, save the current figure
                plt.savefig(buf, format=settings.VISUALIZATION_FORMAT, dpi=settings.VISUALIZATION_DPI)
                
            # Reset buffer position
            buf.seek(0)

            # Get the image data
            image_data = buf.getvalue()
            
            # after you have image_data in bytes
            with open("/app/visual_debug/last_plot.png", "wb") as f:
                f.write(image_data)
            logger.info("Wrote debug plot to /app/visual_debug/last_plot.png")

            with open("/app/visual_debug/last_code.py", "w") as f:
                    f.write(code)
            logger.info("Wrote visualization code to /app/visual_debug/last_code.py")


            # Print debug info
            logger.info(f"Generated visualization with size: {len(image_data)} bytes")
            
            # Return the image data
            return image_data, settings.VISUALIZATION_FORMAT
        
        except Exception as e:
            logger.error(f"Error executing visualization code: {e}", exc_info=True)
            # Create a simple error visualization
            return self._create_error_visualization_image(str(e)), settings.VISUALIZATION_FORMAT
    
    def _create_error_visualization_image(self, error_message: str) -> bytes:
        """
        Create an error visualization image
        
        Args:
            error_message: Error message to display
            
        Returns:
            Image data as bytes
        """
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error creating visualization:\n\n{error_message}", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=14, color='darkred')
        plt.axis('off')
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format=settings.VISUALIZATION_FORMAT, dpi=settings.VISUALIZATION_DPI)
        buf.seek(0)
        
        return buf.getvalue()
    
    def _generate_no_data_visualization(self, message: str) -> Dict[str, Any]:
        """
        Generate a visualization indicating no data is available
        
        Args:
            message: Message to display
            
        Returns:
            Dictionary with visualization data
        """
        logger.info(f"Generating no-data visualization with message: {message}")
        
        # Simple code to create a text-based visualization
        code = f"""
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.text(0.5, 0.5, "{message}", 
         horizontalalignment='center', verticalalignment='center',
         transform=plt.gca().transAxes, fontsize=16)
plt.axis('off')

# Save to buffer
buf = buffer  # Use the buffer provided in the execution environment
plt.savefig(buf, format='{settings.VISUALIZATION_FORMAT}', dpi={settings.VISUALIZATION_DPI})
buf.seek(0)
"""
        
        # Create the visualization
        image_data, image_format = self._execute_visualization_code(code, [])
        
        # Encode image as base64 for transmission
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        return {
            "image_data": base64_image,
            "image_type": f"image/{image_format}",
            "chart_type": "message",
            "explanation": "No data available for visualization"
        }
    
    def _generate_error_visualization(self, error_message: str) -> Dict[str, Any]:
        """
        Generate a visualization indicating an error occurred
        
        Args:
            error_message: Error message to display
            
        Returns:
            Dictionary with visualization data
        """
        logger.info(f"Generating error visualization with message: {error_message}")
        
        # Simple code to create an error visualization
        code = f"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Error creating visualization:\\n\\n{error_message}", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=14, wrap=True,
                color='darkred')
        plt.axis('off')

        # Save to buffer
        buf = buffer  # Use the buffer provided in the execution environment
        plt.savefig(buf, format='{settings.VISUALIZATION_FORMAT}', dpi={settings.VISUALIZATION_DPI})
        buf.seek(0)
        """
        
        try:
            # Create the visualization
            image_data, image_format = self._execute_visualization_code(code, [])
            
            # Encode image as base64 for transmission
            base64_image = base64.b64encode(image_data).decode('utf-8')
            logger.info(f"Generated error visualization with base64 length: {len(base64_image)}")
            
            return {
                "image_data": base64_image,
                "image_type": f"image/{image_format}",
                "chart_type": "error",
                "explanation": f"Error: {error_message}"
            }
        except Exception as e:
            # Last resort - return a minimal response if even the error visualization fails
            logger.error(f"Failed to create error visualization: {e}")
            return {
                "image_data": None,
                "image_type": None,
                "chart_type": None,
                "explanation": f"Visualization failed: {error_message}"
            }