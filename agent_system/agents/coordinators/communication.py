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
        self.sql_agent = SQLAgent()  # SQL agent for database queries
        
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
            
            # Step 2: Find recipients based on recipient_query
            recipient_description = plan.get("recipient_query", "")
            recipients = self._find_recipients(recipient_description, intermediate_steps)
            
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
    
    def _find_recipients(self, recipient_description: str, intermediate_steps: List[Dict[str, Any]]) -> List[str]:
        """
        Find recipients based on natural language description
        
        This method sends natural language queries to the SQL agent to find recipients.
        It first tries targeted queries based on the recipient description, with fallbacks
        as needed.
        
        Args:
            recipient_description: Natural language description of recipients
            intermediate_steps: List to record intermediate steps
            
        Returns:
            List of recipient email addresses
        """
        recipients = []
        logger.info(f"Starting to find recipients for: {recipient_description}")
        
        # Step 1: Prepare queries based on description type
        queries = []
        
        # Check what type of recipients we're looking for
        is_probation = "probation" in recipient_description.lower() or "academic standing" in recipient_description.lower()
        is_financial_aid = "financial aid" in recipient_description.lower() or "scholarship" in recipient_description.lower()
        is_department = "department" in recipient_description.lower() or "program" in recipient_description.lower()
        is_gpa = "gpa" in recipient_description.lower() or "grade" in recipient_description.lower()
        
        # Step 2: Formulate appropriate queries for the recipient type
        
        if is_probation:
            # For academic probation, first explore academic standing values
            queries.append("Find all distinct values in the AcademicStanding column of the PsStudentAcademicRecord table")
            
            # Use a more flexible query that handles different possible academic standing values
            queries.append("Find email addresses of all students whose AcademicStanding contains 'Probation' or is exactly 'Probation'")
            
            # Add a GPA-based fallback
            queries.append("Find email addresses of all students with a GPA below 2.5")
            
        elif is_financial_aid:
            # For financial aid, explore financial aid status values
            queries.append("Find all distinct financial aid status values available in the database")
            
            # Then find students with specific statuses
            queries.append("Find email addresses of all students who have received financial aid")
            
        elif is_department:
            # For department queries, find departments and then students in them
            queries.append("Find all available departments or programs in the database")
            
            # Extract department from description if available
            department_query = f"Find email addresses of all students in the {recipient_description}"
            queries.append(department_query)
            
        elif is_gpa:
            # For GPA-related queries
            queries.append("Find email addresses of all students with GPA below 2.5")
            queries.append("Find email addresses of all students with GPA above 3.5")
            
        else:
            # Generic student query
            queries.append("Find email addresses of all current students")
        
        # Step 3: Execute queries in sequence until we find recipients
        for query in queries:
            # Skip empty queries
            if not query:
                continue
                
            logger.info(f"Executing query: {query}")
            
            # Add the query to intermediate steps
            intermediate_steps.append({
                "agent": "sql_agent",
                "action": "query_recipients",
                "input": query,
                "output": "Processing query to find recipients",
                "timestamp": self._get_timestamp()
            })
            
            # Execute the query using the SQL agent
            sql_result = self.sql_agent(query)
            
            # Debug logging for sql result
            if "is_error" in sql_result:
                logger.info(f"Query error status: {sql_result['is_error']}")
            
            if "results" in sql_result:
                logger.info(f"Query returned {len(sql_result['results'])} results")
            
            # Parse the result to find emails
            if not sql_result.get("is_error", True) and sql_result.get("results"):
                results = sql_result.get("results", [])
                logger.info(f"Processing {len(results)} rows from query result")
                
                # Extract potential email addresses from results
                emails_found = 0
                for row in results:
                    # Detailed logging for each row
                    logger.debug(f"Processing row: {row}")
                    for key, value in row.items():
                        if isinstance(value, str) and "@" in value:
                            recipients.append(value)
                            emails_found += 1
                            logger.debug(f"Found email in column {key}: {value}")
                
                logger.info(f"Extracted {emails_found} email addresses from query results")
                
                # If we found emails, we can stop querying
                if recipients:
                    logger.info(f"Found {len(recipients)} recipients with query: {query}")
                    
                    # Record success in intermediate steps
                    intermediate_steps.append({
                        "agent": "sql_agent",
                        "action": "find_recipients",
                        "input": query,
                        "output": f"Found {len(recipients)} recipients",
                        "timestamp": self._get_timestamp()
                    })
                    
                    return recipients
                else:
                    logger.warning(f"Query returned results but no email addresses were found")
            else:
                # Log specific error message if available
                if "error" in sql_result:
                    logger.warning(f"Query error: {sql_result['error']}")
        
        # Step 4: Try one direct GPA query as a last resort
        last_resort_query = """
        SELECT "Person"."EmailAddress"
        FROM "Person"
        JOIN "PsStudentAcademicRecord" ON "Person"."PersonId" = "PsStudentAcademicRecord"."PersonId"
        WHERE "PsStudentAcademicRecord"."GPA" < 2.5;
        """
        
        logger.info("Trying last resort direct GPA query")
        
        try:
            # Execute the query directly using SQL agent's raw_query method if available
            if hasattr(self.sql_agent, 'execute_raw_query'):
                direct_result = self.sql_agent.execute_raw_query(last_resort_query)
            else:
                # Fall back to regular query method with the raw SQL
                direct_result = self.sql_agent(f"Execute this exact SQL query: {last_resort_query}")
            
            # Log the result
            logger.info(f"Last resort query result: {direct_result}")
            
            # Process the results directly
            if "results" in direct_result and direct_result["results"]:
                for row in direct_result["results"]:
                    for value in row.values():
                        if isinstance(value, str) and "@" in value:
                            recipients.append(value)
                
                logger.info(f"Last resort query found {len(recipients)} email addresses")
                
                if recipients:
                    return recipients
        except Exception as e:
            logger.error(f"Error executing last resort query: {e}")
        
        # Step 5: If all queries failed to find recipients, use fallback
        logger.warning("No recipients found with database queries, using fallbacks")
        
        fallback_recipients = []
        
        if is_probation:
            fallback_recipients = ["academic_support@university.edu"]
        elif is_financial_aid:
            fallback_recipients = ["financial_aid_students@university.edu"]
        elif is_department:
            fallback_recipients = ["departmental_students@university.edu"]
        else:
            fallback_recipients = ["all_students@university.edu"]
        
        logger.info(f"Using fallback recipients: {fallback_recipients}")
        
        # Record fallback in intermediate steps
        intermediate_steps.append({
            "agent": "communication",
            "action": "use_fallback_recipients",
            "input": recipient_description,
            "output": f"Using fallback recipients: {fallback_recipients}",
            "timestamp": self._get_timestamp()
        })
        
        return fallback_recipients
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string"""
        return datetime.now().isoformat()