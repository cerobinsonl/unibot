import logging
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List

from config import settings, AGENT_CONFIGS, get_llm
from agents.specialists.sql_agent import SQLAgent
from agents.specialists.synthetic_agent import SyntheticAgent

class SyntheticDataCoordinator:
    """
    Synthetic Data Coordinator handles complex multi‑table synthetic data
    generation.  Each run uses its own temp‑table prefix to isolate results.
    """

    # 1. Plan the spec: tables, fields, relationships, constraints
    PLANNING_PROMPT = """
You are the Synthetic Data Coordinator for a university administrative system.
Your job is to break a high‑level request into a JSON specification for synthetic data.

Database schema:
{schema_info}

Temporary‑table prefix to use for this run: {temp_prefix}

User request:
{user_input}

Respond ONLY in JSON with keys:
- tables: map table name → number of rows to generate
- fields: map qualified field name → value rules (e.g. distributions, choices)
- relationships: map child_column → "ParentTable.ParentKey"
- constraints: list of global constraints (e.g. "max 50 students per class")
"""

    # 2. Validate results: simple SQL checks
    VALIDATION_PROMPT = """
You are the Synthetic Data Coordinator.
Your generated data spec was applied but these validation checks failed:

{validation_errors}

Please suggest adjustments to the spec JSON so that these constraints will pass,
and output only the revised spec JSON.
"""

    # 3. Synthesize a human‑friendly summary
    SYNTHESIS_PROMPT = """
You have successfully generated synthetic data according to the spec.
Summary of what was created:
- Classes: {class_count}
- Students: {student_count} (GPA μ={gpa_mean:.2f}, σ={gpa_std:.2f})
- Enrollments: {enrollment_count} (max per class = {max_per_class})

Write a concise confirmation for the user.
"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.llm = get_llm("synthetic_data_coordinator")
        self.sql_agent = SQLAgent()
        self.synthetic_agent = SyntheticAgent()

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input: str = state.get("user_input", "")
        steps: List[Dict[str, Any]] = state.setdefault("intermediate_steps", [])

        # 0. Generate a unique temp‑table prefix
        run_ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        temp_prefix = f"temp_synth_{run_ts}_{suffix}"

        # 1. Planning: get JSON spec from LLM
        plan_prompt = self.PLANNING_PROMPT.format(
            schema_info=self.sql_agent.schema_info,
            temp_prefix=temp_prefix,
            user_input=user_input
        )
        self.logger.debug("Planning prompt:\n" + plan_prompt)
        plan_resp = self.llm.invoke(plan_prompt).content.strip()
        spec = json.loads(plan_resp)

        steps.append({
            "agent": "synthetic_data",
            "action": "create_spec",
            "input": user_input,
            "output": spec,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Inject our prefix
        spec["temp_prefix"] = temp_prefix

        # 2. Delegate to SyntheticAgent
        gen_result = self.synthetic_agent(spec)
        steps.append({
            "agent": "synthetic_agent",
            "action": "generate_data",
            "input": spec,
            "output": gen_result,
            "timestamp": datetime.utcnow().isoformat()
        })

        # 3. Validation: run simple SQL checks
        validation_errors = []
        # Example: check classes count
        class_cnt = gen_result.get("created", {}).get("Class", 0)
        cnt_res = self.sql_agent(f"SELECT COUNT(*) AS cnt FROM \"{temp_prefix}_Class\";")
        actual_class_cnt = cnt_res.get("results", [{}])[0].get("cnt", 0)
        if actual_class_cnt != class_cnt:
            validation_errors.append(f"Class count mismatch: expected {class_cnt}, got {actual_class_cnt}")

        # Example: check max enrollment per class
        enroll_res = self.sql_agent(
            f"SELECT MAX(count) AS maxcnt FROM ("
            f"  SELECT ClassId, COUNT(*) AS count"
            f"  FROM \"{temp_prefix}_Enrollment\""
            f"  GROUP BY ClassId"
            f") sub;"
        )
        max_per_class = enroll_res.get("results", [{}])[0].get("maxcnt", 0)
        if "max_per_class" in spec.get("constraints", []):
            # you could parse that constraint; here we assume <=50
            if max_per_class > 50:
                validation_errors.append(f"Some classes have >50 students (max {max_per_class})")

        if validation_errors:
            # 3a. Auto‑correct spec once
            val_prompt = self.VALIDATION_PROMPT.format(
                validation_errors="\n".join(validation_errors)
            )
            self.logger.debug("Validation correction prompt:\n" + val_prompt)
            corrected = self.llm.invoke(val_prompt).content.strip()
            spec_corrected = json.loads(corrected)
            spec_corrected["temp_prefix"] = temp_prefix

            steps.append({
                "agent": "synthetic_data",
                "action": "auto_correct",
                "input": validation_errors,
                "output": spec_corrected,
                "timestamp": datetime.utcnow().isoformat()
            })

            # Retry generation once
            gen_result = self.synthetic_agent(spec_corrected)
            steps.append({
                "agent": "synthetic_agent",
                "action": "generate_data_retry",
                "input": spec_corrected,
                "output": gen_result,
                "timestamp": datetime.utcnow().isoformat()
            })

        # 4. Synthesis: build confirmation
        # Pull stats back via SQLAgent
        sc = self.sql_agent(f"SELECT COUNT(*) AS cnt FROM \"{temp_prefix}_Class\";")
        pc = self.sql_agent(f"SELECT COUNT(*) AS cnt, AVG(GPA) AS avg, STDDEV(GPA) AS std FROM \"{temp_prefix}_Person\";")
        ec = self.sql_agent(f"SELECT COUNT(*) AS cnt FROM \"{temp_prefix}_Enrollment\";")

        class_count = sc["results"][0]["cnt"]
        student_count = pc["results"][0]["cnt"]
        gpa_mean = pc["results"][0]["avg"] or 0
        gpa_std  = pc["results"][0]["std"] or 0
        enrollment_count = ec["results"][0]["cnt"]

        synth_prompt = self.SYNTHESIS_PROMPT.format(
            class_count=class_count,
            student_count=student_count,
            gpa_mean=gpa_mean,
            gpa_std=gpa_std,
            enrollment_count=enrollment_count,
            max_per_class=max_per_class
        )
        confirmation = self.llm.invoke(synth_prompt).content.strip()

        steps.append({
            "agent": "synthetic_data",
            "action": "synthesize_confirmation",
            "input": {
                "class_count": class_count,
                "student_count": student_count,
                "gpa_mean": gpa_mean,
                "gpa_std": gpa_std,
                "enrollment_count": enrollment_count
            },
            "output": confirmation,
            "timestamp": datetime.utcnow().isoformat()
        })

        # 5. Return to user
        state["response"] = confirmation
        state["intermediate_steps"] = steps
        state["current_agent"] = "synthetic_data"
        return state
