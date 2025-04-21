import json
import logging
import re
from typing import Dict, Any, List

from config import get_llm
from agents.specialists.sql_agent import SQLAgent
from agents.specialists.analysis_agent import AnalysisAgent
from agents.specialists.visualization_agent import VisualizationAgent

logger = logging.getLogger(__name__)


class DataAnalysisCoordinator:
    """
    Coordinator pipeline:
      1.  LLM‑produced JSON plan (SQL / analysis / viz)
      2.  Execute SQL via SQLAgent
      3.  Optional deep analysis (guarded so we do **not** re‑aggregate data that
          has already been aggregated by the SQL step)
      4.  Optional visualization
      5.  Synthesize a concise answer for the Director
    """

    # NOTE: added guideline: if SQL already aggregates, analysis_task should be empty
    PLANNING_PROMPT = """
You are the Data Analysis Coordinator in a university administrative system.
Produce a JSON plan with keys:
  "sql_task": string,
  "analysis_task": string,   # leave **blank** if the SQL already returns aggregated results
  "visualization_task": string,
  "needs_visualization": boolean
Respond with raw JSON only (no code fences).
Database schema:
{schema_info}
User request: {user_input}
"""

    SYNTHESIS_PROMPT = """
Synthesize the results for the user.

Request: {user_input}
SQL Query: {sql_query}
Sample data: {sql_sample}
Analysis summary: {analysis_summary}
Visualization attached: {has_chart}
"""

    # --- initialise helpers -------------------------------------------------
    def __init__(self):
        self.llm = get_llm("data_analysis_coordinator")
        self.sql_agent = SQLAgent()
        self.analysis_agent = AnalysisAgent()
        self.visualization_agent = VisualizationAgent()

    # ------------------------------------------------------------------------
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry‑point used by LangGraph."""
        user_input: str = state.get("user_input", "")

        # 1. Ask the LLM for a plan ------------------------------------------------
        raw_plan = self.llm.invoke(
            self.PLANNING_PROMPT.format(
                schema_info=self.sql_agent.schema_info,
                user_input=user_input,
            )
        ).content
        clean_plan = re.sub(r"^```(?:json)?\s*", "", raw_plan, flags=re.MULTILINE)
        clean_plan = re.sub(r"\s*```$", "", clean_plan, flags=re.MULTILINE)
        try:
            plan = json.loads(clean_plan)
            logger.debug("Plan: %s", plan)
        except json.JSONDecodeError:
            state.update(
                response="Sorry, I couldn't interpret the analysis plan.",
                current_agent="data_analysis",
            )
            return state

        # 2. Run the SQL -----------------------------------------------------------
        sql_res = self.sql_agent(plan.get("sql_task", ""))
        if sql_res.get("is_error"):
            state.update(
                response=f"SQL error: {sql_res.get('error')}",
                current_agent="data_analysis",
            )
            return state

        # Helper to decide if SQL results are already aggregated ---------------
        def _looks_aggregated(column_names: List[str]) -> bool:
            agg_tokens = ("count", "sum", "avg", "total", "min", "max", "median")
            return all(
                col.lower() == "department" or any(tok in col.lower() for tok in agg_tokens)
                for col in column_names
            )

        is_aggregated = _looks_aggregated(sql_res.get("column_names", []))

        # 3. Deep analysis (optional) --------------------------------------------
        wants_analysis = "analysis" in user_input.lower() or bool(plan.get("analysis_task"))
        do_analysis = wants_analysis and not is_aggregated
        analysis_summary = ""
        analysis_failed_msg = ""
        if do_analysis and plan.get("analysis_task"):
            analysis_out = self.analysis_agent(
                {
                    "task": plan["analysis_task"],
                    "data": sql_res.get("results", []),
                    "column_names": sql_res.get("column_names", []),
                    "row_count": sql_res.get("row_count", 0),
                }
            )
            if analysis_out.get("is_error"):
                analysis_failed_msg = analysis_out.get("error", "analysis error")
                logger.warning("Analysis failed: %s", analysis_failed_msg)
            else:
                analysis_summary = analysis_out.get("summary", "")

        # 4. Visualization (optional) -------------------------------------------
        wants_chart = (
            "chart" in user_input.lower()
            or "plot" in user_input.lower()
            or plan.get("needs_visualization", False)
        )
        viz = None
        if wants_chart and plan.get("visualization_task"):
            viz = self.visualization_agent(
                {
                    "task": plan["visualization_task"],
                    "data": sql_res.get("results", []),
                    "column_names": sql_res.get("column_names", []),
                    "analysis": {"summary": analysis_summary},
                }
            )

        # 5. Craft final text ------------------------------------------------------
        sample = sql_res.get("results", [])[:3]
        final_text = self.llm.invoke(
            self.SYNTHESIS_PROMPT.format(
                user_input=user_input,
                sql_query=sql_res.get("query", ""),
                sql_sample=sample,
                analysis_summary=(analysis_summary if analysis_summary else "No further analysis required."),
                has_chart="Yes" if viz else "No",
            )
        ).content

        if analysis_failed_msg:
            final_text += f"\n\n(Note: deeper statistical analysis was skipped – {analysis_failed_msg})"

        # 6. Update conversation state -------------------------------------------
        state["response"] = final_text
        state["current_agent"] = "data_analysis"
        state.setdefault("intermediate_steps", []).append(
            {
                "agent": "data_analysis",
                "output": {
                    "sql": sql_res,
                    "analysis": analysis_summary if analysis_summary else None,
                    "analysis_error": analysis_failed_msg if analysis_failed_msg else None,
                    "visualization": bool(viz),
                },
            }
        )
        if viz:
            state["visualization"] = viz
        return state