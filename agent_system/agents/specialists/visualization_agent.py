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

IMPORTANT: 
- Use ONLY the data provided to you
- Do NOT assume or generate data that doesn't exist in the input
- If the data is empty or has very few records, create a simple message visualization stating "No data available" or "Insufficient data"
- Use ONLY these libraries: matplotlib, seaborn, pandas, numpy

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
            
            logger.info(f"Creating visualization with task: {task[:100]}...")
            logger.info(f"Data has {len(data)} records and {len(column_names)} columns")
            
            # Check if data is empty or insufficient
            if not data or len(data) < 1:
                logger.warning(f"Insufficient data for visualization: {len(data) if data else 0} records")
                return self._generate_no_data_visualization("No data available for visualization.")
            
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
                    logger.info("Attempted extraction using import pattern")
                
                chart_type = "unknown"
                explanation = "Visualization code extracted from non-JSON response"
            
            if not code:
                logger.warning("No visualization code could be extracted from the response")
                return self._generate_no_data_visualization("Couldn't generate appropriate visualization code.")
            
            # Log the code being used
            logger.debug(f"Visualization code: {code[:500]}...")
            
            # Create the visualization using the extracted code
            logger.info("Executing visualization code...")
            image_data, image_format = create_visualization(code, data)
            
            # Check if we have valid image data
            if not image_data or len(image_data) == 0:
                logger.error("No image data returned from create_visualization")
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
        image_data, image_format = create_visualization(code, [])
        
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
            image_data, image_format = create_visualization(code, [])
            
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