from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

# Import configuration
from config import settings, AGENT_CONFIGS, get_llm

# Import specialists
from agents.specialists.email_agent import EmailAgent
from agents.specialists.sql_agent import SQLAgent

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
        self.sql_agent = SQLAgent()  # Add SQL agent for database queries
        
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
- recipient_query: A natural language description of who should receive this communication
- subject: Subject line for the communication
- content: The content of the message
- priority: "high", "medium", or "low"

Example:
{{
  "communication_type": "email",
  "recipient_query": "Get email addresses of all students who have applied for financial aid but haven't completed their application",
  "subject": "Important Update About Final Exams",
  "content": "Dear Students, This is to inform you that the final exam schedule has been updated...",
  "priority": "high"
}}

Important: Make sure the content is appropriate for a university setting and formatted correctly for the chosen communication type.

User request: {user_input}
"""
        
        # Create a query construction prompt that emphasizes exploration
        self.recipient_query_prompt = """
You need to find the appropriate recipients for a university communication.

Request: {recipient_query}

Based on this request, create a natural language query for our SQL Agent to find the right recipients in our database. 
The SQL Agent will translate your natural language query into SQL.

IMPORTANT: For fields that might have unknown values (like Academic Standing or Status fields),
instruct the SQL Agent to first explore those values and then construct an appropriate query, rather than assuming
specific values exist in the database.

For example, instead of:
"Find students with AcademicStanding = 'Academic Probation'"

Prefer:
"First, find all possible values in the AcademicStanding column of the PsStudentAcademicRecord table, then find
all students whose AcademicStanding indicates academic difficulty, looking for values like 'Probation',
'Warning', or any value containing words about academic concerns."

Your response should be a clear step-by-step instruction that:
1. First explores relevant field values if needed
2. Then constructs a query that will find the appropriate recipients
3. Uses broad matching conditions rather than assuming exact values

Your response:
"""
        
        # Create a two-step exploratory query prompt
        self.exploratory_query_prompt = """
In order to find recipients for a university communication, we need to approach this in two steps:

Step 1: Explore the database's relevant field values to understand what's available.
Step 2: Construct a targeted query based on those actual values.

For a request about: {recipient_type}

Please provide two sequential queries:
1. First query: Explore the relevant field values in the database to understand what actual values exist
2. Second query: Find the appropriate recipients based on the actual values that would be revealed by the first query

For example, if finding students on academic probation:
1. First query: "Find all distinct values in the AcademicStanding column of the PsStudentAcademicRecord table"
2. Second query: "Based on the values found, find all students whose AcademicStanding indicates probation or academic difficulty (likely values containing terms like 'probation', 'warning', 'concern', etc.)"

Please generate these two sequential queries for our current request.
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
2. Indicate who it was sent to (in general terms and include the recipient count)
3. Mention if it was successfully delivered
4. Avoid using placeholder text like "[Financial Aid Office]" - use "Financial Aid Office" without brackets
5. Be professional and concise, as appropriate for university administrative staff

User request: {user_input}

Communication details: 
Type: {comm_type}
Recipients: {recipient_count} recipients ({recipient_description})
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
                
                recipient_query_match = re.search(r'"recipient_query"\s*:\s*"([^"]+)"', planning_response)
                recipient_query = recipient_query_match.group(1) if recipient_query_match else "Get email addresses of all students"
                
                priority_match = re.search(r'"priority"\s*:\s*"([^"]+)"', planning_response)
                priority = priority_match.group(1) if priority_match else "medium"
                
                plan = {
                    "communication_type": comm_type,
                    "recipient_query": recipient_query,
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
            
            # Step 2: Convert recipient query to natural language that the SQL Agent can understand
            recipient_description = plan.get("recipient_query", "")
            recipients = []
            
            # Determine if this requires special field exploration
            needs_exploration = False
            recipient_type = ""
            if "probation" in recipient_description.lower() or "probation" in user_input.lower():
                needs_exploration = True
                recipient_type = "students on academic probation"
            elif "standing" in recipient_description.lower() or "gpa" in recipient_description.lower():
                needs_exploration = True
                recipient_type = "students with specific academic standing"
            elif "status" in recipient_description.lower() and ("application" in recipient_description.lower() 
                                                            or "financial" in recipient_description.lower()):
                needs_exploration = True
                recipient_type = "students with specific financial aid status"
            
            if recipient_description:
                if needs_exploration:
                    # Use exploratory approach with multiple queries
                    exploration_prompt = self.exploratory_query_prompt.format(recipient_type=recipient_type)
                    exploration_plan = self.llm.invoke(exploration_prompt).content
                    
                    # Log the exploration plan
                    logger.info(f"Using exploratory approach for {recipient_type}")
                    logger.info(f"Exploration plan: {exploration_plan}")
                    
                    # Extract the exploration query
                    exploration_query = None
                    query_lines = exploration_plan.split("\n")
                    for i, line in enumerate(query_lines):
                        if "first query" in line.lower() and i+1 < len(query_lines):
                            exploration_query = query_lines[i+1].strip('"\'')
                            break
                    
                    if not exploration_query and "1." in exploration_plan:
                        # Try to find the query after a numbered list item
                        parts = exploration_plan.split("1.")
                        if len(parts) > 1:
                            exploration_query = parts[1].strip().split("\n")[0].strip('"\'')
                    
                    if not exploration_query:
                        # Fallback to a generic exploration query based on recipient type
                        if "probation" in recipient_type.lower():
                            exploration_query = "Find all distinct values in the AcademicStanding column of the PsStudentAcademicRecord table"
                        elif "financial" in recipient_type.lower():
                            exploration_query = "Find all distinct values in the Status column of the FinancialAid table"
                    
                    # Log and execute the exploration query if we have one
                    if exploration_query:
                        logger.info(f"Executing exploration query: {exploration_query}")
                        
                        # Execute the exploration query
                        exploration_result = self.sql_agent(exploration_query)
                        
                        # Add exploration step to intermediate steps
                        intermediate_steps.append({
                            "agent": "sql_agent",
                            "action": "explore_field_values",
                            "input": exploration_query,
                            "output": "Exploration results",
                            "timestamp": self._get_timestamp()
                        })
                        
                        # Extract possible values for matching
                        possible_values = []
                        if exploration_result and not exploration_result.get("is_error", False):
                            results = exploration_result.get("results", [])
                            
                            # Try to extract distinct values
                            for row in results:
                                for key, value in row.items():
                                    if value and isinstance(value, str):
                                        possible_values.append(value)
                        
                        # Log the possible values found
                        logger.info(f"Exploration found values: {possible_values}")
                        
                        # Extract the main query based on exploration results
                        recipient_query = None
                        for i, line in enumerate(query_lines):
                            if "second query" in line.lower() and i+1 < len(query_lines):
                                recipient_query = query_lines[i+1].strip('"\'')
                                break
                        
                        if not recipient_query and "2." in exploration_plan:
                            # Try to find the query after a numbered list item
                            parts = exploration_plan.split("2.")
                            if len(parts) > 1:
                                recipient_query = parts[1].strip().split("\n")[0].strip('"\'')
                        
                        # Modify the recipient query to include actual values if we found them
                        if recipient_query and possible_values:
                            # Check if it's a probation query
                            if "probation" in recipient_type.lower() or "academic" in recipient_type.lower():
                                values_str = ", ".join([f"'{v}'" for v in possible_values])
                                recipient_query += f" (Found values: {values_str})"
                    
                    # If we couldn't construct a good query from exploration
                    if not recipient_query:
                        # Generate a better query with the exploration results
                        enhanced_prompt = self.recipient_query_prompt.format(recipient_query=recipient_description)
                        recipient_query = self.llm.invoke(enhanced_prompt).content
                else:
                    # Standard approach for simple queries
                    enhanced_prompt = self.recipient_query_prompt.format(recipient_query=recipient_description)
                    recipient_query = self.llm.invoke(enhanced_prompt).content
                
                # Log the final natural language query
                logger.info(f"Natural language query: {recipient_query}")
                
                # Use SQL agent for the actual query
                query_result = self.sql_agent(recipient_query)
                
                # Add SQL execution to intermediate steps
                intermediate_steps.append({
                    "agent": "sql_agent",
                    "action": "query_recipients",
                    "input": recipient_query,
                    "output": "SQL result with recipient emails",  # Don't log all emails for privacy
                    "timestamp": self._get_timestamp()
                })
                
                # Extract email addresses
                if query_result and not query_result.get("is_error", False):
                    results = query_result.get("results", [])
                    
                    # Try to find email addresses in results
                    for row in results:
                        # Look for email column
                        email_value = None
                        for key, value in row.items():
                            # Check if this column looks like an email field
                            if isinstance(value, str) and ("email" in key.lower() or "@" in value):
                                email_value = value
                                break
                                
                        if email_value:
                            recipients.append(email_value)
            
            # If no recipients found or query failed, use a fallback
            if not recipients:
                # Log the issue
                logger.warning("No recipients found from database query, using fallback")
                
                # Try one more direct approach if it was an academic standing or status query
                if needs_exploration:
                    # Make one more attempt with a different approach
                    fallback_query = None
                    if "probation" in recipient_type.lower():
                        fallback_query = "Find email addresses of all students with a GPA below 2.5"
                    elif "financial" in recipient_type.lower():
                        fallback_query = "Find email addresses of all students who have any record in the FinancialAid table"
                    
                    if fallback_query:
                        logger.info(f"Attempting fallback query: {fallback_query}")
                        fallback_result = self.sql_agent(fallback_query)
                        
                        # Add fallback step to intermediate steps
                        intermediate_steps.append({
                            "agent": "sql_agent",
                            "action": "fallback_query",
                            "input": fallback_query,
                            "output": "SQL fallback result",
                            "timestamp": self._get_timestamp()
                        })
                        
                        # Extract emails from fallback
                        if fallback_result and not fallback_result.get("is_error", False):
                            results = fallback_result.get("results", [])
                            
                            # Extract emails from results
                            for row in results:
                                for key, value in row.items():
                                    if isinstance(value, str) and ("email" in key.lower() or "@" in value):
                                        recipients.append(value)
                                        
                            if recipients:
                                logger.info(f"Found {len(recipients)} recipients with fallback query")
                
                # If still no recipients, use general fallback
                if not recipients:
                    if "financial aid" in user_input.lower():
                        recipients = ["financial_aid_students@university.edu"]
                        recipient_description = "students eligible for financial aid"
                    elif "faculty" in user_input.lower():
                        recipients = ["all_faculty@university.edu"]
                        recipient_description = "faculty members"
                    elif "probation" in user_input.lower():
                        recipients = ["academic_support@university.edu"]
                        recipient_description = "students on academic probation"
                    else:
                        recipients = ["all_students@university.edu"]
                        recipient_description = "all students"
            
            # Step 3: Handle the communication based on type
            result = None
            if plan["communication_type"] == "email":
                # Use the email agent
                result = self.email_agent({
                    "recipients": recipients,
                    "subject": plan["subject"],
                    "content": plan["content"],
                    "priority": plan["priority"]
                })
                
                # Add email step to intermediate steps
                intermediate_steps.append({
                    "agent": "email_agent",
                    "action": "send_email",
                    "input": {
                        "recipients_count": len(recipients),
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
                    "message": f"Notification queued for {len(recipients)} recipients",
                    "notification_id": f"notif-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                }
                
                # Add notification step to intermediate steps
                intermediate_steps.append({
                    "agent": "notification_agent",
                    "action": "send_notification",
                    "input": {
                        "recipients_count": len(recipients),
                        "content": "Notification content"  # Don't log full content for privacy
                    },
                    "output": result,
                    "timestamp": self._get_timestamp()
                })
                
            elif plan["communication_type"] == "sms":
                # Mock SMS for now (would use an SMSAgent in production)
                result = {
                    "status": "success",
                    "message": f"SMS queued for {len(recipients)} recipients",
                    "sms_id": f"sms-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                }
                
                # Add SMS step to intermediate steps
                intermediate_steps.append({
                    "agent": "sms_agent",
                    "action": "send_sms",
                    "input": {
                        "recipients_count": len(recipients),
                        "content": "SMS content"  # Don't log full content for privacy
                    },
                    "output": result,
                    "timestamp": self._get_timestamp()
                })
            
            # Step 4: Synthesize results
            synthesis_input = {
                "user_input": user_input,
                "comm_type": plan["communication_type"],
                "recipient_count": len(recipients),
                "recipient_description": recipient_description,
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