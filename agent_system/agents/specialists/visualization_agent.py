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
    Visualization Agent generates Python visualization code via LLM,
    executes it to produce a base64-encoded image, and returns it.
    """

    # Prompt to generate visualization code
    PLANNING_PROMPT = """
You are a Visualization Agent in a university administration system.
Generate a Python function named `create_chart(df, buf)` that:
1. Creates an appropriate chart for the task.
2. Uses matplotlib or seaborn.
3. Sets titles, labels, and ensures clarity.
4. Saves the figure to `buf` via `plt.savefig(buf, format='png')` and `buf.seek(0)`.

Task description: {task_description}
Columns available: {column_list}
"""

    def __init__(self):
        self.llm = get_llm("visualization_agent")

    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        df = pd.DataFrame(input_data.get('data', []))
        task_desc = input_data.get('task', '')
        cols = input_data.get('column_names', list(df.columns))

        # Generate visualization code via LLM
        prompt = self.PLANNING_PROMPT.format(
            task_description=task_desc,
            column_list=cols
        )
        code_resp = self.llm.invoke(prompt).content
        logger.debug("Raw visualization LLM output:\n%s", code_resp)


        # Extract Python code from response (and strip any leading language tag)
        code = code_resp
        if code.strip().startswith("```"):
            parts = code.split('```')
            code = parts[1] if len(parts) >= 3 else parts[-1]

        # Remove a leading “python” line if present
        code = re.sub(r'^\s*python\s*\n', '', code, flags=re.IGNORECASE)
        
        logger.debug("Cleaned visualization code snippet:\n%s", code)

        # Execute generated code
        buf = io.BytesIO()
        exec_globals = {
            'pd': pd,
            'np': np,
            'df': df,
            'buf': buf,
            'plt': __import__('matplotlib.pyplot')
        }

        logger.debug("Executing visualization code with globals %s", list(exec_globals.keys()))
        logger.debug("Code preview:\n%s", "\n".join(code.splitlines()[:10]))
        try:
            # 1) Define create_chart(df, buf)
            exec(code, exec_globals)
            # 2) Now call it
            chart_fn = exec_globals.get('create_chart')
            if not callable(chart_fn):
                raise ValueError("No create_chart() function defined.")
            chart_fn(df, buf)
            img_bytes = buf.getvalue()
            if not img_bytes:
                raise ValueError("No image data produced by visualization code.")
            b64 = base64.b64encode(img_bytes).decode('utf-8')
            return {'is_error': False, 'image_data': b64, 'image_type': 'image/png'}
        except Exception as e:
            logger.error("[VisualizationAgent] Execution error: %s", e)
            return {'is_error': True, 'error': str(e)}