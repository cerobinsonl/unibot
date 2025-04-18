import logging
import re
import json
from typing import Dict, Any, List
from config import get_llm

logger = logging.getLogger(__name__)

class SyntheticAgent:
    """
    Synthetic Data Generator Agent.

    Takes a validated JSON specification (fields, relationships, constraints, etc.)
    and produces two lists:
      - ddl: SQL statements to create temporary tables
      - dml: SQL INSERT statements to populate them
    """

    # Prompt template for spec → SQL conversion
    SPEC_TO_SQL_PROMPT = """
You are the Synthetic Data Generator. Using this validated JSON spec:
{spec}

Produce **only** JSON with keys:
- ddl: [ <CREATE TABLE ... for temp tables>, … ]
- dml: [ <INSERT … statements to populate>, … ]
"""

    def __init__(self):
        # Initialize the LLM for this agent
        self.llm = get_llm("synthetic_agent")

    def __call__(self, spec: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate DDL and DML based on the provided spec.

        Args:
            spec: A dict containing the user‐validated specification, e.g.:
              {
                "fields": { ... },
                "relationships": { ... },
                "constraints": [ ... ]
              }

        Returns:
            A dict with:
              - ddl: List of CREATE TABLE statements (for temp tables)
              - dml: List of INSERT statements to populate them
        """
        try:
            # Serialize the spec neatly
            spec_str = json.dumps(spec, indent=2)

            # Build and invoke the prompt
            prompt = self.SPEC_TO_SQL_PROMPT.format(spec=spec_str)
            response = self.llm.invoke(prompt).content

            # Strip any Markdown code fences
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"```$", "", cleaned)

            # Parse the JSON output
            payload = json.loads(cleaned)
            ddl_statements = payload.get("ddl", [])
            dml_statements = payload.get("dml", [])

            return {
                "ddl": ddl_statements,
                "dml": dml_statements
            }

        except Exception as e:
            logger.error("Error in SyntheticAgent: %s", e, exc_info=True)
            # Return the error in a consistent format
            return {
                "ddl": [],
                "dml": [],
                "error": str(e)
            }
