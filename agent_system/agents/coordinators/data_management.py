import logging
import json
from datetime import datetime
from typing import Dict, Any, List

from config import settings, AGENT_CONFIGS, get_llm
from agents.specialists.sql_agent import SQLAgent

class DataManagementCoordinator:
    """
    Data Management Coordinator handles INSERT/UPDATE/DELETE operations
    by delegating all SQL generation and execution to SQLAgent.
    """

    # Prompt to plan the data operation, schema will be injected dynamically
    PLANNING_PROMPT = """
You are the Data Management Coordinator for a university administrative system.
Your responsibility is to oversee database operations: INSERT, UPDATE, DELETE.

Here is the database schema:
{schema_info}

User request: {user_input}

Please format your plan as JSON with keys:
- operation_type: "insert", "update", or "delete"
- table: Name of the table to operate on
- data: Object of column→value pairs (for insert/update)
- condition: WHERE clause description (for update/delete)
- validation_rules: List of any data integrity checks to perform
"""

    # Prompt to correct a failing operation instruction based on schema and error
    CORRECTION_PROMPT = """
You are the Data Management Coordinator for a university administrative system.
Your last operation instruction caused this SQL error:
{error}

Original instruction:
"{operation_nl}"

Database schema:
{schema_info}

Please produce a corrected natural‑language instruction (only the corrected instruction),
so that SQLAgent can generate and execute a valid SQL statement.
"""

    # Prompt to synthesize the final confirmation message
    SYNTHESIS_PROMPT = """
You are the Data Management Coordinator for a university administrative system.
You have just performed a database operation.

User request:
{user_input}

Operation details:
- Type: {operation_type}
- Table: {table}
- Affected records: {affected_records}
- Result message: {result_message}

Please write a concise confirmation message summarizing what was done.
"""

    def __init__(self):
        self.llm = get_llm("data_management_coordinator")
        self.sql_agent = SQLAgent()
        self.logger = logging.getLogger(__name__)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input: str = state.get("user_input", "")
        steps: List[Dict[str, Any]] = state.setdefault("intermediate_steps", [])

        self.logger.info(f"DataManagementCoordinator start: {user_input}")

        # 1. Plan the operation, injecting schema
        schema_info = self.sql_agent.schema_info
        plan_prompt = self.PLANNING_PROMPT.format(
            schema_info=schema_info,
            user_input=user_input
        )
        self.logger.debug(f"Planning prompt:\n{plan_prompt}")
        plan_response = self.llm.invoke(plan_prompt).content
        self.logger.debug(f"Plan response:\n{plan_response}")

        # Strip any markdown fences
        plan_text = plan_response.strip()
        if plan_text.startswith("```"):
            plan_text = "\n".join(plan_text.split("\n")[1:]).rsplit("\n```", 1)[0]

        try:
            plan = json.loads(plan_text)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse plan JSON", exc_info=True)
            state["response"] = "Sorry, I couldn't understand how to perform that operation."
            state["current_agent"] = "data_management"
            return state

        steps.append({
            "agent": "data_management",
            "action": "create_plan",
            "input": user_input,
            "output": plan,
            "timestamp": datetime.now().isoformat()
        })

        op_type = plan.get("operation_type")
        table = plan.get("table")
        data = plan.get("data", {})
        condition = plan.get("condition", "")
        if op_type not in ("insert", "update", "delete") or not table:
            self.logger.error(f"Invalid plan: {plan}")
            state["response"] = "Plan was invalid. Please check the operation type and table name."
            state["current_agent"] = "data_management"
            return state

        # 2. Build the natural‑language instruction to SQLAgent
        if op_type == "insert":
            nl_instruction = (
                f"Insert into {table} with values: {json.dumps(data)}"
            )
        elif op_type == "update":
            nl_instruction = (
                f"Update {table}, set {json.dumps(data)} where {condition}"
            )
        else:  # delete
            nl_instruction = f"Delete from {table} where {condition}"

        # 3. Try & possibly auto‑correct
        sql_result = None
        for attempt in range(2):
            self.logger.info(f"SQLAgent attempt {attempt+1}: {nl_instruction}")
            sql_result = self.sql_agent(nl_instruction)
            self.logger.info(f"SQLAgent result: {sql_result}")

            if not sql_result.get("is_error", False):
                break

            # On error, ask LLM to correct the NL instruction
            error_msg = sql_result.get("error", "Unknown error")
            corr_prompt = self.CORRECTION_PROMPT.format(
                error=error_msg,
                operation_nl=nl_instruction,
                schema_info=schema_info
            )
            self.logger.debug(f"Correction prompt:\n{corr_prompt}")
            corrected = self.llm.invoke(corr_prompt).content.strip()
            self.logger.info(f"Corrected instruction: {corrected}")

            steps.append({
                "agent": "data_management",
                "action": "auto_correct",
                "input": {"original": nl_instruction, "error": error_msg},
                "output": corrected,
                "timestamp": datetime.now().isoformat()
            })
            nl_instruction = corrected

        # 4. Handle persistent failure
        if sql_result.get("is_error", False):
            self.logger.error("Operation failed after retry")
            state["response"] = (
                f"I’m sorry, I couldn’t complete the {op_type} on {table}. "
                f"Error: {sql_result.get('error')}"
            )
            state["current_agent"] = "data_management"
            return state

        # 5. Record execution step
        steps.append({
            "agent": "data_management",
            "action": "execute_operation",
            "input": nl_instruction,
            "output": sql_result,
            "timestamp": datetime.now().isoformat()
        })

        # 6. Determine affected rows
        affected = sql_result.get("affected_rows")
        if affected is None:
            # fallback to row_count if provided
            affected = sql_result.get("row_count", 0)

        result_msg = sql_result.get("message", f"{affected} rows affected")

        # 7. Synthesize confirmation
        synth_prompt = self.SYNTHESIS_PROMPT.format(
            user_input=user_input,
            operation_type=op_type,
            table=table,
            affected_records=affected,
            result_message=result_msg
        )
        self.logger.debug(f"Synthesis prompt:\n{synth_prompt}")
        confirmation = self.llm.invoke(synth_prompt).content
        self.logger.info(f"Confirmation: {confirmation}")

        # 8. Update state and return
        state["response"] = confirmation
        state["intermediate_steps"] = steps
        state["current_agent"] = "data_management"
        return state

    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string"""
        return datetime.now().isoformat()