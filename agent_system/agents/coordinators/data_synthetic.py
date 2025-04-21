from __future__ import annotations

"""SyntheticDataCoordinator – updated to avoid .format KeyError
Located at **agent_system/agents/coordinators/data_synthetic.py**
"""

import json, re, ast, logging
import logging
import re
from datetime import datetime
from typing import Any, Dict, List

from config import get_llm
from agents.specialists.synthetic_agent import SyntheticAgent
from agents.specialists.sql_agent import SQLAgent

logger = logging.getLogger(__name__)


class SyntheticDataCoordinator:
    """Plan + delegate synthetic data generation via SyntheticAgent."""

    # curly braces inside text need doubling to survive str.format()
    PLANNING_PROMPT = (
        """
You are the Synthetic Data Coordinator for a university administrative system.
Break down the user request into a JSON spec with keys:
  "tables": object  // map "Table" -> rows
  "fields": object  // map "Table.Column" -> rule string
  "relationships": object  // map "ChildTable.ChildFK" -> "ParentTable.ParentPK"
  "constraints": array  // optional global constraints

Use only these rule patterns (case‑insensitive):
   fake.<provider>     e.g. fake.first_name
   choice(a,b,c)
   int_range(min,max)
   normal(mu,sigma)
   constant(value)
Do not output “uuid”, “firstName”, “date”, or any other custom keyword.

If you need a UUID, use fake.uuid4
If you need a date of birth, use fake.date

Primary‑key rule:
* For any column that is a PRIMARY KEY (or ends with “Id”), either
  - omit the column entirely (so Postgres generates it with SERIAL/IDENTITY), **or**
  - set its rule to "fake.uuid4" to guarantee uniqueness.

Respond **only** with the JSON object.

Database schema:
{schema}

User request:
{user_input}
"""
    )

    def __init__(self):
        self.llm = get_llm("data_synthetic_coordinator")
        self.generator = SyntheticAgent()

    # ---------------------------------------------------------------------
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input: str = state.get("user_input", "")
        steps: List[Dict[str, Any]] = state.setdefault("intermediate_steps", [])

        # 1. Ask the LLM for a generation spec ---------------------------------
        schema_info = self.generator.schema_info
        prompt = self.PLANNING_PROMPT.format(schema=schema_info, user_input=user_input)
        raw = self.llm.invoke(prompt).content.strip()

        # 1. Remove outer quotes if the model wrapped the whole block
        if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
            raw = raw[1:-1]

        # 2. Strip ```json fences (start or end may still have spaces/newlines)
        clean = re.sub(r"^```\\w*\\s*", "", raw, flags=re.MULTILINE)
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE)
        clean = re.sub(r'^\s*json\s*\n', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r"\\s*```$", "", clean, flags=re.MULTILINE).strip()

        logger.debug("Cleaned LLM output:\n%s", clean)

        # 3. Try strict JSON
        try:
            # 3‑0. parse → spec  --------------------------------------------
            spec = json.loads(clean)

            # ---------------------------------------------------------------
            # 3‑1. ►►  NORMALISE & SANITISE THE SPEC  ◄◄
            # ---------------------------------------------------------------
            # 3‑1a. ensure each tables[table] is an int
            fixed_tables = {}
            for tbl, val in spec.get("tables", {}).items():
                if isinstance(val, int):
                    fixed_tables[tbl] = val
                elif isinstance(val, dict) and "rows" in val:
                    fixed_tables[tbl] = int(val["rows"])
                elif isinstance(val, list):
                    fixed_tables[tbl] = len(val)
                else:
                    # instead of dropping the table entirely, give it a 0 count
                    logger.warning("Unrecognized row spec for %s: %r — defaulting to 0 rows", tbl, val)
                    fixed_tables[tbl] = 0
            spec["tables"] = fixed_tables

            # ---------------------------------------------------------------
            # 3‑1b. ►►  NORMALISE PK RULES FOR INTEGER IDs  ◄◄
            # ---------------------------------------------------------------
            # For any *Id PK* field, pick an int_range that starts above the current max in the DB
            sql_agent = SQLAgent()

            for fq_col in list(spec.get("fields", {})):
                table, col = fq_col.split(".", 1)
                if col.lower().endswith("id"):
                    # how many rows we’re going to generate
                    N = spec["tables"].get(table, 0)
                    # ask the real table what its current max is
                    resp = sql_agent(f'SELECT MAX("{col}") AS max_id FROM "{table}";')
                    current_max = resp["results"][0].get("max_id") or 0
                    start = current_max + 1
                    end   = current_max + N
                    spec["fields"][fq_col] = f"int_sequence(start={start})"

        except json.JSONDecodeError:
            # 4. Lenient: replace common Pythonisms → JSON
            safe = (
                clean.replace("true", "true")
                    .replace("false", "false")
                    .replace("None", "null")
            )
            try:
                spec = ast.literal_eval(safe)
            except Exception:
                logger.error("Unable to parse spec:\n%s", clean, exc_info=True)
                state.update(
                    response="I couldn't understand the synthetic‑data spec the LLM produced.",
                    current_agent="synthetic_data",
                )
                return state


        # --- NEW: enforce num_rows constraints first --------------------------
        for c in spec.get("constraints", []):
            m = re.match(r'num_rows\((\w+)\)\s*==\s*(\d+)', c)
            if m:
                table, cnt = m.group(1), int(m.group(2))
                spec["tables"][table] = cnt

        # --- NEW: fallback to user_input count if still all zeros ------------
        if all(v == 0 for v in spec["tables"].values()):
            m = re.search(r'(\d+)', user_input)
            if m:
                count = int(m.group(1))
                # pick out only the real (non‑temp) tables
                real_tables = [t for t in spec["tables"].keys() if not t.startswith("temp_")]
                # assign that count to *each* table
                for tbl in real_tables:
                    spec["tables"][tbl] = count

        logger.debug("Specs :\n%s", spec)

        # Ensure a temp prefix
        spec["temp_prefix"] = spec.get(
            "temp_prefix", f"temp_synth_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        )
        
        logger.debug("Specs failed:\n%s", spec)

        # 2. Delegate to SyntheticAgent ----------------------------------------
        try:
            report = self.generator(spec)  # returns dict {table: rows_inserted}
            logger.debug("Synthetic Agent response:\n%s", report)
        except Exception as e:
            logger.exception("SyntheticAgent failed")
            state.update(
                response=f"Synthetic data generation failed: {e}",
                current_agent="synthetic_data",
            )
            return state

        # 3. Compose user‑facing confirmation ----------------------------------
        tables_done = ", ".join(f"{t} ({n})" for t, n in report.items())
        confirmation = (
            f"Generated synthetic data successfully. Rows inserted per table: {tables_done}."
        )

        state.update(
            response=confirmation,
            current_agent="synthetic_data",
        )
        return state
