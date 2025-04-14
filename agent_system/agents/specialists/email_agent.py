import logging
from typing import Dict, List, Any, Optional
import os
import json
from datetime import datetime

# Import configuration
from config import settings, AGENT_CONFIGS, get_llm

# Configure logging
logger = logging.getLogger(__name__)

class EmailAgent:
    """
    Email Agent is responsible for formatting and sending emails to university
    stakeholders like students, faculty, and staff.
    """
    
    def __init__(self):
        """Initialize the Email Agent"""
        # Create the LLM using the helper function
        self.llm = get_llm("email_agent")
        
        # Create the email formatting prompt
        self.formatting_prompt = """
You are the Email Agent for a university administrative system.
Your role is to format and send emails to university stakeholders.

You need to format an email communication for university stakeholders. Your task is to:

1. Format the content in a clear, professional manner appropriate for university communications
2. Include appropriate greeting and closing
3. Ensure the tone matches the priority level and audience
4. Add a university signature block at the end

Format your response as a JSON object with these keys:
- formatted_subject: The revised subject line
- formatted_content: The full email body with greeting, content, and signature
- suggestions: Any suggestions for improving communication effectiveness

Example:
{
  "formatted_subject": "Important Update: Final Exam Schedule Changes",
  "formatted_content": "Dear Students,\\n\\nI hope this email finds you well. I'm writing to inform you about important changes to the final exam schedule...\\n\\nSincerely,\\n\\nDr. Jane Smith\\nAcademic Affairs Office\\nUniversity Name\\nemail@university.edu",
  "suggestions": ["Consider sending a follow-up reminder one week before exams", "Include a link to the full exam schedule"]
}

Recipients: {recipients}
Subject: {subject}
Content: {content}
Priority: {priority}

Please format this into a professional university email.
"""
    
    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format and send an email
        
        Args:
            input_data: Dictionary containing email details
            
        Returns:
            Dictionary with send status and metadata
        """
        try:
            # Extract information from input
            recipients = input_data.get("recipients", [])
            subject = input_data.get("subject", "University Communication")
            content = input_data.get("content", "")
            priority = input_data.get("priority", "medium")
            
            # Convert recipients to string for prompt if it's a list
            recipients_str = ", ".join(recipients) if isinstance(recipients, list) else recipients
            
            # Format the email
            formatted_prompt = self.formatting_prompt.format(
                recipients=recipients_str,
                subject=subject,
                content=content,
                priority=priority
            )
            
            formatting_response = self.llm.invoke(formatted_prompt).content
            
            # Extract formatted content
            try:
                formatted = json.loads(formatting_response)
                formatted_subject = formatted.get("formatted_subject", subject)
                formatted_content = formatted.get("formatted_content", content)
                suggestions = formatted.get("suggestions", [])
            except json.JSONDecodeError:
                # If not valid JSON, use original with minimal formatting
                formatted_subject = subject
                formatted_content = f"Dear Recipients,\n\n{content}\n\nBest regards,\nUniversity Administration"
                suggestions = []
            
            # In a real implementation, this would send the actual email
            # For the POC, we'll just simulate sending
            if os.getenv("MOCK_EMAIL_SENDING", "true").lower() == "true":
                # Mock sending
                message_id = f"<{datetime.now().strftime('%Y%m%d%H%M%S')}.{hash(formatted_content) % 10000}@university.edu>"
                send_status = "success"
            else:
                # Real implementation would be here
                # Example:
                # send_result = send_email(recipients, formatted_subject, formatted_content)
                # message_id = send_result.message_id
                # send_status = send_result.status
                raise NotImplementedError("Real email sending not implemented yet")
            
            # Return the results
            return {
                "status": send_status,
                "message": f"Email sent to {len(recipients) if isinstance(recipients, list) else 1} recipient(s)",
                "message_id": message_id,
                "subject": formatted_subject,
                "suggestions": suggestions
            }
            
        except Exception as e:
            logger.error(f"Error in Email Agent: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to send email: {str(e)}",
                "message_id": None,
                "subject": subject if 'subject' in locals() else None
            }