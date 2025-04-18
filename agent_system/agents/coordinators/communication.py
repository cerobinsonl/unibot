import logging
import json
from datetime import datetime
from typing import Dict, Any, List
import re
from config import settings, AGENT_CONFIGS, get_llm
from agents.specialists.email_agent import EmailAgent
from agents.specialists.sql_agent import SQLAgent

class CommunicationCoordinator:
    """
    Communication Coordinator manages all messaging and notification tasks
    by delegating to specialized messaging agents, with schema‑aware planning
    and automatic retry on SQL errors using only LLM capabilities.
    """
    # Prompt to plan the communication, schema will be injected dynamically
    PLANNING_PROMPT = """
You are the Communication Coordinator for a university administrative system.
Your role is to handle all messaging and notification related tasks.

You need to create a plan for handling this communication request. Analyze what type of communication is needed:

1. Email - Formal communication to students, faculty, or staff
2. Notification - System notifications to be shown in the university portal
3. SMS - Urgent messages that need immediate attention

Format your response as a JSON object with these keys:
- communication_type: "email", "notification", or "sms"
- recipient_query: A natural language description of who should receive this communication
- subject: Subject line for the communication
- content: The content of the message (use HTML tags for line breaks and lists)
- priority: "high", "medium", or "low"

User request: {user_input}
"""

    # Prompt to ask the LLM to correct a failing recipient_query based on schema and error
    CORRECTION_PROMPT = """
You are the Communication Coordinator for a university administrative system.
Your last recipient_query caused this SQL error:
{error}

The original recipient_query was:
"{original_query}"

Here is the database schema:
{schema_info}

Please generate a corrected recipient_query (just the value for "recipient_query") that will work
with the schema above. Respond with only the corrected recipient_query string.
"""

    # Prompt to synthesize the final confirmation message
    SYNTHESIS_PROMPT = """
You are the Communication Coordinator for a university administrative system.
Your role is to confirm to the user what was done.

User request:
{user_input}

Communication details:
- Type: {comm_type}
- Recipients: {recipient_count} ({recipient_list})
- Subject: {subject}
- Email agent result: {result}

Please write a concise confirmation message summarizing the action taken.
"""

    def __init__(self):
        self.llm = get_llm("communication_coordinator")
        self.sql_agent = SQLAgent()
        self.email_agent = EmailAgent()
        self.logger = logging.getLogger(__name__)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        steps: List[Dict[str, Any]] = state.setdefault("intermediate_steps", [])

        self.logger.info(f"Coordinator start: {user_input}")

        # 1. Plan the communication, injecting schema
        schema_info = self.sql_agent.schema_info
        self.logger.info("Injecting schema into planning prompt")
        plan_prompt = (
            self.PLANNING_PROMPT
            + "\n\nDatabase schema:\n"
            + schema_info
        ).format(user_input=user_input)
        self.logger.info(f"Plan prompt:\n{plan_prompt}")
        plan_response = self.llm.invoke(plan_prompt).content
        self.logger.info(f"Plan response:\n{plan_response}")

        # Strip Markdown fences before parsing
        plan_text = re.sub(r"^```(?:json)?\s*", "", plan_response, flags=re.MULTILINE)
        plan_text = re.sub(r"\s*```$", "", plan_text, flags=re.MULTILINE)

        try:
            plan = json.loads(plan_text)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse plan JSON after stripping fences", exc_info=True)
            state["response"] = "Sorry, I couldn't understand the plan from the LLM."
            state["current_agent"] = "communication"
            return state

        steps.append({
            "agent": "communication",
            "action": "create_plan",
            "input": user_input,
            "output": plan,
            "timestamp": datetime.now().isoformat()
        })

        recipient_query = plan["recipient_query"]

        # 2. Try the SQL query, with up to one auto‑correction pass
        sql_result = None
        for attempt in range(2):
            self.logger.info(f"SQLAgent attempt {attempt+1} for query: {recipient_query}")
            sql_result = self.sql_agent(recipient_query)
            self.logger.info(f"SQLAgent result: {sql_result}")

            if not sql_result.get("is_error", False):
                break

            # Auto‑correct via LLM
            error_msg = sql_result.get("error", "Unknown error")
            correction_prompt = self.CORRECTION_PROMPT.format(
                error=error_msg,
                original_query=recipient_query,
                schema_info=schema_info
            )
            self.logger.info(f"Correction prompt:\n{correction_prompt}")
            corrected = self.llm.invoke(correction_prompt).content.strip()
            self.logger.info(f"Corrected recipient_query: {corrected}")

            steps.append({
                "agent": "communication",
                "action": "auto_correct",
                "input": {"original_query": recipient_query, "error": error_msg},
                "output": corrected,
                "timestamp": datetime.now().isoformat()
            })
            recipient_query = corrected

        # 3. Extract email addresses or apply fallback
        recipients: List[str] = []
        if sql_result and not sql_result.get("is_error") and sql_result.get("results"):
            for row in sql_result["results"]:
                for v in row.values():
                    if isinstance(v, str) and "@" in v:
                        recipients.append(v)
            self.logger.info(f"Extracted recipients: {recipients}")
        else:
            self.logger.warning("No valid recipients found, using fallback address")
            recipients = ["all_students@university.edu"]

        steps.append({
            "agent": "communication",
            "action": "extract_recipients",
            "input": sql_result.get("results") if sql_result else None,
            "output": f"Recipients: {recipients}",
            "timestamp": datetime.now().isoformat()
        })

        # 4. Send the email (only for email type)
        email_result: Dict[str, Any] = {}
        if plan["communication_type"] == "email":
            self.logger.info(f"Sending email to {len(recipients)} recipients")
            email_input = {
                "recipients": recipients,
                "subject": plan["subject"],
                "content": plan["content"],
                "priority": plan["priority"]
            }
            self.logger.info(f"EmailAgent input: {email_input}")
            email_result = self.email_agent(email_input)
            self.logger.info(f"EmailAgent result: {email_result}")
            steps.append({
                "agent": "email_agent",
                "action": "send_email",
                "input": {"recipients_count": len(recipients), "subject": plan["subject"]},
                "output": email_result,
                "timestamp": datetime.now().isoformat()
            })
        else:
            # (Implement notification/SMS similarly if needed)
            self.logger.info(f"No email sent, communication_type={plan['communication_type']}")
            email_result = {"message": "No email sent for this communication type."}

        # 5. Synthesize final confirmation
        synth_prompt = self.SYNTHESIS_PROMPT.format(
            user_input=user_input,
            comm_type=plan["communication_type"],
            recipient_count=len(recipients),
            recipient_list=", ".join(recipients),
            subject=plan["subject"],
            result=email_result.get("message", "")
        )
        self.logger.info(f"Synthesis prompt:\n{synth_prompt}")
        confirmation = self.llm.invoke(synth_prompt).content
        self.logger.info(f"Final confirmation: {confirmation}")

        state["response"] = confirmation
        state["intermediate_steps"] = steps
        state["current_agent"] = "communication"
        return state


    def _correct_query(self, original_query: str, error: str) -> str:
        """
        Ask the LLM to rewrite the recipient_query to match the schema, given an error.
        """
        schema_info = self.sql_agent.schema_info
        correction_prompt = (
            AGENT_CONFIGS['communication_coordinator']['system_prompt']
            + "\nDatabase schema:\n" + schema_info
            + f"\nOriginal query: {original_query}"
            + f"\nError: {error}"
            + "\nPlease provide a corrected recipient_query in JSON, e.g.: {\"recipient_query\": \"…\"}."
        )
        correction_response = self.llm.invoke(correction_prompt).content
        try:
            corrected = json.loads(correction_response).get('recipient_query', original_query)
        except json.JSONDecodeError:
            corrected = original_query
        return corrected

    def _ts(self) -> str:
        return datetime.now().isoformat()
