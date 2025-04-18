import logging
import json
import re
from datetime import datetime
from typing import Dict, Any, List

from config import get_llm
from agents.specialists.synthetic_agent import SyntheticAgent
from agents.specialists.sql_agent import SQLAgent

class SyntheticDataCoordinator:
    """
    Synthetic Data Coordinator handles complex synthetic data generation
    workflows: planning distributions, validating against schema,
    executing DDL/DML, and confirming results.
    """

    # 1) Initial spec planning & insight prompt
    PLANNING_INSIGHT_PROMPT = """
You are the Synthetic Data Coordinator for a university administrative system.
Your job is to translate a user request into a JSON specification for synthetic data generation.

DATABASE SCHEMA:
{schema_info}

USER REQUEST:
{user_input}

Please reply with **only** JSON with keys:
- tables: {{ "<TableName>": <row_count>, … }}
- distributions: {{ "<TableName>.<ColumnName>": {{ "min":…, "max":…, "mean":…, "std":… }}, … }}
- relationships: {{ "<ChildTable>.<FKColumn>": "<ParentTable>.<PKColumn>", … }}
- constraints: [ … ]   # any global rules, e.g. "Each Class enrolls ≤ 50 students"
"""

    # 2) Spec correction prompt
    SPEC_CORRECTION_PROMPT = """
The JSON spec you provided did not validate against the current database contents.

Original spec:
{original_spec}

Validation results:
{validation_results}

Please reply with a corrected JSON spec only.
"""

    # 3) Generation prompt
    GENERATION_PROMPT = """
You are the Synthetic Data Generator. Using this validated JSON spec:
{spec}

Produce **only** JSON with keys:
- ddl: [ <CREATE TABLE ... for temp tables>, … ]
- dml: [ <INSERT … statements to populate>, … ]
"""

    # 4) DML correction prompt
    DML_CORRECTION_PROMPT = """
Some DDL/DML statements failed with error:
{error}

Original failed statement:
{failed_stmt}

Please provide a corrected version of that single statement only.
"""

    # 5) Final synthesis prompt
    SYNTHESIS_PROMPT = """
Synthetic data generation complete.

User request:
{user_input}

Records created:
{creation_summary}

Write a concise confirmation message summarizing these results.
"""

    def __init__(self):
        self.llm = get_llm("synthetic_data_coordinator")
        self.synthetic_agent = SyntheticAgent()
        self.sql_agent = SQLAgent()
        self.logger = logging.getLogger(__name__)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        steps: List[Dict[str, Any]] = state.setdefault("intermediate_steps", [])
        schema_info = self.sql_agent.schema_info

        self.logger.info(f"SyntheticDataCoordinator start: {user_input}")

        # --- Step 1: Plan & insight ---
        plan_prompt = self.PLANNING_INSIGHT_PROMPT.format(
            schema_info=schema_info,
            user_input=user_input
        )
        self.logger.debug(f"Planning prompt:\n{plan_prompt}")
        plan_resp = self.llm.invoke(plan_prompt).content.strip()
        # Clean Markdown fences if any
        plan_text = re.sub(r"^```json\s*|\s*```$", "", plan_resp, flags=re.MULTILINE)
        try:
            spec = json.loads(plan_text)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse spec JSON", exc_info=True)
            state["response"] = "Sorry, I couldn't understand how to generate synthetic data for that request."
            state["current_agent"] = "synthetic_data"
            return state

        steps.append({
            "agent": "synthetic_data",
            "action": "plan_spec",
            "input": user_input,
            "output": spec,
            "timestamp": datetime.now().isoformat()
        })

        # --- Step 2: Validate spec via SQLAgent ---
        validation_errors = []
        for table, expected_count in spec.get("tables", {}).items():
            q = f'SELECT COUNT(*) AS cnt FROM "{table}"'
            res = self.sql_agent(q)
            actual = res.get("results", [{}])[0].get("cnt", 0)
            if abs(actual - expected_count) > max(1, 0.05 * expected_count):
                validation_errors.append(
                    f'Table {table}: expected {expected_count}, actual {actual}'
                )

        if validation_errors:
            val_prompt = self.SPEC_CORRECTION_PROMPT.format(
                original_spec=json.dumps(spec, indent=2),
                validation_results=json.dumps(validation_errors, indent=2)
            )
            self.logger.debug(f"Spec correction prompt:\n{val_prompt}")
            corrected = self.llm.invoke(val_prompt).content.strip()
            corrected_text = re.sub(r"^```json\s*|\s*```$", "", corrected, flags=re.MULTILINE)
            try:
                spec = json.loads(corrected_text)
            except json.JSONDecodeError:
                self.logger.error("Failed to parse corrected spec JSON", exc_info=True)
                state["response"] = "I attempted to adjust the plan but couldn't produce a valid spec."
                state["current_agent"] = "synthetic_data"
                return state

            steps.append({
                "agent": "synthetic_data",
                "action": "correct_spec",
                "input": validation_errors,
                "output": spec,
                "timestamp": datetime.now().isoformat()
            })

        # --- Step 3: Generate DDL + DML ---
        gen_prompt = self.GENERATION_PROMPT.format(spec=json.dumps(spec, indent=2))
        self.logger.debug(f"Generation prompt:\n{gen_prompt}")
        gen_resp = self.llm.invoke(gen_prompt).content.strip()
        gen_text = re.sub(r"^```json\s*|\s*```$", "", gen_resp, flags=re.MULTILINE)
        try:
            ddl_dml = json.loads(gen_text)
            ddls = ddl_dml.get("ddl", [])
            dmls = ddl_dml.get("dml", [])
        except json.JSONDecodeError:
            self.logger.error("Failed to parse DDL/DML JSON", exc_info=True)
            state["response"] = "I couldn't generate the SQL statements for synthetic data."
            state["current_agent"] = "synthetic_data"
            return state

        steps.append({
            "agent": "synthetic_data",
            "action": "generate_ddl_dml",
            "input": spec,
            "output": ddl_dml,
            "timestamp": datetime.now().isoformat()
        })

        # --- Step 4: Execute DDL + DML with retry on error ---
        all_statements = ddls + dmls
        for stmt in all_statements:
            res = self.sql_agent(stmt)
            if res.get("is_error"):
                # Attempt single-statement correction
                corr_prompt = self.DML_CORRECTION_PROMPT.format(
                    error=res["error"],
                    failed_stmt=stmt
                )
                self.logger.debug(f"DML correction prompt:\n{corr_prompt}")
                corrected_stmt = self.llm.invoke(corr_prompt).content.strip()
                corrected_stmt = re.sub(r"^```sql\s*|\s*```$", "", corrected_stmt, flags=re.MULTILINE)

                steps.append({
                    "agent": "synthetic_data",
                    "action": "correct_ddl_dml",
                    "input": {"original": stmt, "error": res["error"]},
                    "output": corrected_stmt,
                    "timestamp": datetime.now().isoformat()
                })

                # Retry
                res = self.sql_agent(corrected_stmt)
                if res.get("is_error"):
                    self.logger.error(f"Failed to execute statement even after correction: {res['error']}")
                    state["response"] = f"Error populating synthetic data: {res['error']}"
                    state["current_agent"] = "synthetic_data"
                    return state

        steps.append({
            "agent": "synthetic_data",
            "action": "execute_ddl_dml",
            "input": all_statements,
            "output": {"status": "all statements executed"},
            "timestamp": datetime.now().isoformat()
        })

        # --- Step 5: Final validation & summary collection ---
        creation_summary = {}
        for table in spec.get("tables", {}):
            temp = f"temp_{table.lower()}"
            q = f'SELECT COUNT(*) AS cnt FROM "{temp}"'
            res = self.sql_agent(q)
            cnt = res.get("results", [{}])[0].get("cnt", 0)
            creation_summary[temp] = cnt

        steps.append({
            "agent": "synthetic_data",
            "action": "final_validation",
            "input": None,
            "output": creation_summary,
            "timestamp": datetime.now().isoformat()
        })

        # --- Step 6: Synthesize confirmation ---
        summary_lines = [f"{tbl}: {cnt} rows" for tbl, cnt in creation_summary.items()]
        summary_text = "\n".join(summary_lines)
        synth_prompt = self.SYNTHESIS_PROMPT.format(
            user_input=user_input,
            creation_summary=summary_text
        )
        self.logger.debug(f"Synthesis prompt:\n{synth_prompt}")
        confirmation = self.llm.invoke(synth_prompt).content.strip()

        state["response"] = confirmation
        state["intermediate_steps"] = steps
        state["current_agent"] = "synthetic_data"
        return state
