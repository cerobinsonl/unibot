import logging
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import traceback

# Import settings
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Set default styling for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def create_visualization(code: str, data: List[Dict[str, Any]]) -> Tuple[bytes, str]:
    """
    Create visualization from code and data
    
    Args:
        code: Python code that generates a matplotlib/seaborn visualization
        data: Data to visualize
        
    Returns:
        Tuple of (image data as bytes, image format)
    """
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
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
        
        # Print debug info
        print(f"Generated visualization with size: {len(image_data)} bytes")
        
        # Return the image data
        return image_data, settings.VISUALIZATION_FORMAT
    
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        logger.error(traceback.format_exc())
        
        # Create a simple error visualization
        return create_error_visualization(str(e)), settings.VISUALIZATION_FORMAT

def create_error_visualization(error_message: str) -> bytes:
    """
    Create a simple visualization indicating an error
    
    Args:
        error_message: Error message to display
        
    Returns:
        Image data as bytes
    """
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, f"Visualization Error:\n\n{error_message}", 
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=14, wrap=True)
    plt.axis('off')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format=settings.VISUALIZATION_FORMAT, dpi=settings.VISUALIZATION_DPI)
    buf.seek(0)
    
    return buf.getvalue()

def create_sample_visualization(title: str = "Sample Visualization") -> Tuple[bytes, str]:
    """
    Create a sample visualization for testing
    
    Args:
        title: Title for the visualization
        
    Returns:
        Tuple of (image data as bytes, image format)
    """
    # Create sample data
    data = {
        'Category': ['A', 'B', 'C', 'D', 'E'],
        'Value1': [5, 7, 3, 9, 4],
        'Value2': [3, 8, 5, 6, 7]
    }
    df = pd.DataFrame(data)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Bar plot
    ax = sns.barplot(x='Category', y='Value1', data=df, color='steelblue', label='Series 1')
    sns.barplot(x='Category', y='Value2', data=df, color='lightcoral', label='Series 2', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format=settings.VISUALIZATION_FORMAT, dpi=settings.VISUALIZATION_DPI)
    buf.seek(0)
    
    return buf.getvalue(), settings.VISUALIZATION_FORMAT

def encode_image_base64(image_data: bytes) -> str:
    """
    Encode image data as base64 string
    
    Args:
        image_data: Image data as bytes
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image_data).decode('utf-8')

def visualization_to_html(image_data: bytes, image_format: str, title: str = None, description: str = None) -> str:
    """
    Convert visualization to HTML for display
    
    Args:
        image_data: Image data as bytes
        image_format: Image format (e.g., 'png', 'svg')
        title: Optional title for the visualization
        description: Optional description
        
    Returns:
        HTML string
    """
    # Encode image as base64
    encoded_image = encode_image_base64(image_data)
    
    # Create HTML
    html = f'<div class="visualization-container">'
    
    if title:
        html += f'<h3>{title}</h3>'
    
    html += f'<img src="data:image/{image_format};base64,{encoded_image}" alt="Visualization">'
    
    if description:
        html += f'<p>{description}</p>'
    
    html += '</div>'
    
    return html