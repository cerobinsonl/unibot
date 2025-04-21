import json
import logging
import base64
import io
from typing import Dict, Any
import re

import pandas as pd
import numpy as np

from config import get_llm

logger = logging.getLogger(__name__)


class AnalysisAgent:
    """Generates and executes task‑specific Python analysis code via LLM."""

    PLANNING_PROMPT = """
You are an Analysis Agent in a university administration system.
Generate a Python function named `perform_analysis(df)` that:
1. Computes relevant statistics for the task:
   - Use mean, median, std, and count for numeric columns.
   - Compute frequency distributions for categorical columns.
2. Returns a JSON‑serializable dict with keys:
   - "summary": brief text summary of findings
   - "stats": mapping column→computed metrics

The function should assign its result to a variable named `analysis_output`.

Task description: {task_description}
"""

    def __init__(self):
        self.llm = get_llm("analysis_agent")

    # ---------------------------------------------------------------------
    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        df = pd.DataFrame(input_data.get("data", []))
        task_desc = input_data.get("task", "")

        # 0. Bail out early if DataFrame empty
        if df.empty:
            return {
                "is_error": True,
                "error": "No data provided for analysis",
            }

        # 1. Ask LLM to write analysis code --------------------------------
        prompt = self.PLANNING_PROMPT.format(task_description=task_desc)
        code_resp = self.llm.invoke(prompt).content
        logger.debug("Raw analysis LLM output:\n%s", code_resp)

        # 2. Extract the code block (strip fences, leading ```python etc.)
        code = code_resp
        if code.strip().startswith("```"):
            parts = code.split("```")
            code = parts[1] if len(parts) >= 3 else parts[-1]
        code = re.sub(r"^\s*python\s*\n", "", code, flags=re.IGNORECASE)
        logger.debug("Cleaned analysis code snippet:\n%s", code)

        # 3. Light static check – ensure referenced columns exist -------------
        col_refs = set(re.findall(r"df\[['\"]([^'\"]+)['\"]", code))
        missing = col_refs - set(df.columns)
        if missing:
            return {
                "is_error": True,
                "error": f"Missing columns in analysis DataFrame: {', '.join(missing)}",
            }

        # 4. Execute the code -------------------------------------------------
        exec_globals = {
            "pd": pd,
            "np": np,
            "df": df,
            "analysis_output": None,
        }
        try:
            exec(code, exec_globals)
            analysis_fn = exec_globals.get("perform_analysis")
            if not callable(analysis_fn):
                raise ValueError("perform_analysis() not defined in generated code")
            exec_globals["analysis_output"] = analysis_fn(df)
            output = exec_globals["analysis_output"]
            if output is None:
                raise ValueError("perform_analysis() returned None")
            return {"is_error": False, **output}
        except Exception as e:
            logger.error("[AnalysisAgent] Execution error: %s", e)
            return {
                "is_error": True,
                "error": str(e),
            }
