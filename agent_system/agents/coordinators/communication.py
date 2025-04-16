from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

# Import configuration
from config import settings, AGENT_CONFIGS, get_llm

# Import specialists
from agents.specialists.email_agent import EmailAgent

# Configure logging
logger = logging.getLogger(__name__)

class CommunicationCoordinator:
    """
    Communication Coordinator manages all messaging and notification tasks
    by delegating to specialized messaging agents.
    """
    
    def __init__(self):
        """Initialize the Communication Coordinator"""
        # Create the LLM using the helper function
        self.llm = get_llm("communication_coordinator")
        
        # Initialize specialist agents
        self.email_agent = EmailAgent()
        
        # Create the task planning prompt
        self.planning_prompt = """
You are the Communication Coordinator for a university administrative system.
Your role is to handle all messaging and notification related tasks.

You need to create a plan for handling this communication request. Analyze what type of communication is needed:

1. Email - Formal communication to students, faculty, or staff
2. Notification - System notifications to be shown in the university portal
3. SMS - Urgent messages that need immediate attention

Format your response as a JSON object with these keys:
- communication_type: "email", "notification", or "sms"
- recipients: List of recipient identifiers (emails, IDs, etc.)
- subject: Subject line for the communication
- content: The content of the message
- priority: "high", "medium", or "low"

Example:
{
  "communication_type": "email",
  "recipients": ["all_students@university.edu"],
  "subject": "Important Update About Final Exams",
  "content": "Dear Students, This is to inform you that the final exam schedule has been updated...",
  "priority": "high"
}

Important: Make sure the content is appropriate for a university setting and formatted correctly for the chosen communication type.

User request: {user_input}
"""
        
        # Create the results synthesis prompt
        self.synthesis_prompt = """
You are the Communication Coordinator for a university administrative system.
Your role is to handle all messaging and notification related tasks.

You are synthesizing the results from communication operations to create a response for the user.

Review the communication request and the results of the sending operation, then create a clear response 
that confirms what was done and provides any relevant details.

Your response should:
1. Confirm what type of communication was sent
2. Indicate who it was sent to (in general terms)
3. Mention if it was successfully delivered
4. Offer to help with anything else

Be professional and concise, as appropriate for university administrative staff.

User request: {user_input}

Communication details: 
Type: {comm_type}
Recipients: {recipients}
Subject: {subject}

Sending result: {result}

Create a response summarizing the action taken.
"""
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the communication request by coordinating specialist agents
        
        Args:
            state: Current state of the conversation
            
        Returns:
            Updated state with communication results
        """
        try:
            # Extract information from state
            user_input = state.get("user_input", "")
            intermediate_steps = state.get("intermediate_steps", [])
            
            # Step 1: Create a plan for handling the request
            formatted_prompt = self.planning_prompt.format(user_input=user_input)
            planning_response = self.llm.invoke(formatted_prompt).content
            
            # Parse the planning response
            try:
                plan = json.loads(planning_response)
            except json.JSONDecodeError:
                # If the response isn't valid JSON, extract what we can
                import re
                
                comm_type_match = re.search(r'"communication_type"\s*:\s*"([^"]+)"', planning_response)
                comm_type = comm_type_match.group(1) if comm_type_match else "email"
                
                subject_match = re.search(r'"subject"\s*:\s*"([^"]+)"', planning_response)
                subject = subject_match.group(1) if subject_match else "University Communication"
                
                content_match = re.search(r'"content"\s*:\s*"([^"]+)"', planning_response)
                content = content_match.group(1) if content_match else user_input
                
                # For recipients, try to extract an array or default to admin
                recipients_match = re.search(r'"recipients"\s*:\s*(\[[^\]]+\])', planning_response)
                if recipients_match:
                    try:
                        recipients = json.loads(recipients_match.group(1))
                    except:
                        recipients = ["admin@university.edu"]
                else:
                    recipients = ["admin@university.edu"]
                
                priority_match = re.search(r'"priority"\s*:\s*"([^"]+)"', planning_response)
                priority = priority_match.group(1) if priority_match else "medium"
                
                plan = {
                    "communication_type": comm_type,
                    "recipients": recipients,
                    "subject": subject,
                    "content": content,
                    "priority": priority
                }
            
            # Add planning step to intermediate steps
            intermediate_steps.append({
                "agent": "communication",
                "action": "create_plan",
                "input": user_input,
                "output": plan,
                "timestamp": self._get_timestamp()
            })
            
            # Step 2: Handle the communication based on type
            result = None
            if plan["communication_type"] == "email":
                # Use the email agent
                result = self.email_agent({
                    "recipients": plan["recipients"],
                    "subject": plan["subject"],
                    "content": plan["content"],
                    "priority": plan["priority"]
                })
                
                # Add email step to intermediate steps
                intermediate_steps.append({
                    "agent": "email_agent",
                    "action": "send_email",
                    "input": {
                        "recipients": plan["recipients"],
                        "subject": plan["subject"],
                        "content": "Email content"  # Don't log full content for privacy
                    },
                    "output": result,
                    "timestamp": self._get_timestamp()
                })
                
            elif plan["communication_type"] == "notification":
                # Mock notification for now (would use a NotificationAgent in production)
                result = {
                    "status": "success",
                    "message": f"Notification queued for {len(plan['recipients'])} recipients",
                    "notification_id": f"notif-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                }
                
                # Add notification step to intermediate steps
                intermediate_steps.append({
                    "agent": "notification_agent",
                    "action": "send_notification",
                    "input": {
                        "recipients": plan["recipients"],
                        "content": "Notification content"  # Don't log full content for privacy
                    },
                    "output": result,
                    "timestamp": self._get_timestamp()
                })
                
            elif plan["communication_type"] == "sms":
                # Mock SMS for now (would use an SMSAgent in production)
                result = {
                    "status": "success",
                    "message": f"SMS queued for {len(plan['recipients'])} recipients",
                    "sms_id": f"sms-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                }
                
                # Add SMS step to intermediate steps
                intermediate_steps.append({
                    "agent": "sms_agent",
                    "action": "send_sms",
                    "input": {
                        "recipients": plan["recipients"],
                        "content": "SMS content"  # Don't log full content for privacy
                    },
                    "output": result,
                    "timestamp": self._get_timestamp()
                })
            
            # Step 3: Synthesize results
            synthesis_input = {
                "user_input": user_input,
                "comm_type": plan["communication_type"],
                "recipients": ", ".join(plan["recipients"]) if isinstance(plan["recipients"], list) else plan["recipients"],
                "subject": plan.get("subject", ""),
                "result": result.get("message", "Message sent successfully.")
            }
            
            formatted_prompt = self.synthesis_prompt.format(**synthesis_input)
            response = self.llm.invoke(formatted_prompt).content
            
            # Update state
            state["response"] = response
            state["intermediate_steps"] = intermediate_steps
            state["current_agent"] = "communication"
            
            return state
            
        except Exception as e:
            logger.error(f"Error in Communication Coordinator: {e}", exc_info=True)
            error_response = f"I encountered an error while processing your communication request: {str(e)}. Please try rephrasing your request or contact support if the issue persists."
            
            # Update state with error
            state["response"] = error_response
            state["current_agent"] = "communication"
            
            return state
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string"""
        return datetime.now().isoformat()