import logging
from typing import Dict, List, Any, Optional
import os
import json
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Import configuration
from config import settings, AGENT_CONFIGS, get_llm

# Configure logging
logger = logging.getLogger(__name__)

class EmailAgent:
    """
    Email Agent is responsible for formatting and sending emails to university
    stakeholders like students, faculty, and staff.
    
    This version integrates with MailHog for testing email sending.
    """
    
    def __init__(self):
        """Initialize the Email Agent"""
        # Create the LLM using the helper function
        self.llm = get_llm("email_agent")
        
        # MailHog configuration
        self.mailhog_server = os.getenv("SMTP_SERVER", "mailhog")
        self.mailhog_port = int(os.getenv("SMTP_PORT", "1025"))
        
        # Email configuration
        self.from_email = os.getenv("FROM_EMAIL", "university-admin@example.edu")
        
        # Print configuration for debugging
        logger.info(f"Email Agent initialized with MailHog: {self.mailhog_server}:{self.mailhog_port}")
        
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
{{
  "formatted_subject": "Important Update: Final Exam Schedule Changes",
  "formatted_content": "Dear Students,\\n\\nI hope this email finds you well. I'm writing to inform you about important changes to the final exam schedule...\\n\\nSincerely,\\n\\nDr. Jane Smith\\nAcademic Affairs Office\\nUniversity Name\\nemail@university.edu",
  "suggestions": ["Consider sending a follow-up reminder one week before exams", "Include a link to the full exam schedule"]
}}

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
            
            # Determine if we should use mock sending or MailHog
            use_mailhog = os.getenv("USE_MAILHOG", "true").lower() == "true"
            
            if use_mailhog:
                # Send to MailHog testing service
                message_id, send_status = self._send_with_mailhog(
                    recipients, 
                    formatted_subject, 
                    formatted_content, 
                    priority
                )
            else:
                # Mock sending
                message_id = f"<{datetime.now().strftime('%Y%m%d%H%M%S')}.{hash(formatted_content) % 10000}@university.edu>"
                send_status = "success"
                self._log_email_details(recipients, formatted_subject, formatted_content, "MOCK", message_id)
            
            # Return the results
            return {
                "status": send_status,
                "message": f"Email sent to {len(recipients) if isinstance(recipients, list) else 1} recipient(s) via MailHog",
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
    
    def _send_with_mailhog(self, recipients, subject, content, priority):
        """
        Send test email using MailHog
        
        Args:
            recipients: List of recipient email addresses
            subject: Email subject
            content: Email content
            priority: Email priority
            
        Returns:
            Tuple of (message_id, status)
        """
        try:
            # Log for debugging
            logger.info(f"Sending email via MailHog to: {recipients}")
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            
            # Convert recipients to list if it's a string
            if isinstance(recipients, str):
                recipients = [recipients]
                
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # Set priority headers
            if priority.lower() == "high":
                msg['X-Priority'] = '1'
                msg['X-MSMail-Priority'] = 'High'
                msg['Importance'] = 'High'
            
            # Add content
            msg.attach(MIMEText(content, 'plain'))
            
            # Connect to MailHog SMTP server
            server = smtplib.SMTP(self.mailhog_server, self.mailhog_port)
            
            # Send email - MailHog doesn't require authentication
            server.sendmail(self.from_email, recipients, msg.as_string())
            server.quit()
            
            # Generate message ID
            message_id = f"<{datetime.now().strftime('%Y%m%d%H%M%S')}.{hash(content) % 10000}@mailhog>"
            status = "success"
            
            # Log the email
            self._log_email_details(recipients, subject, content, "MAILHOG", message_id)
            
            return message_id, status
            
        except Exception as e:
            logger.error(f"MailHog error: {e}")
            return None, f"error: {str(e)}"
    
    def _log_email_details(self, recipients, subject, content, method, message_id):
        """
        Log detailed information about the email for testing purposes
        
        Args:
            recipients: List of recipient email addresses
            subject: Email subject
            content: Email content
            method: Sending method
            message_id: Generated message ID
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log to the application logs
        logger.info(f"EMAIL ({method}): To: {recipients} | Subject: {subject} | ID: {message_id}")
        
        # Print to console for immediate feedback during testing
        print(f"\n===== EMAIL SENT WITH {method} =====")
        print(f"TO: {recipients}")
        print(f"SUBJECT: {subject}")
        print(f"MESSAGE ID: {message_id}")
        print(f"CONTENT PREVIEW: {content[:100]}...")
        print(f"SENDING METHOD: {method}")
        if method == "MAILHOG":
            print("You can view this email at: http://localhost:8025")
        print("===================================\n")