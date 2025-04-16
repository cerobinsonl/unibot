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
                logger.warning("No visualization code could be extracted from the response")
                return self._generate_no_data_visualization("Couldn't generate appropriate visualization code.")
            
            # Log the code being used
            logger.debug(f"Visualization code: {code[:500]}...")
            
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
        logger.info("Generating error visualization")
        
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