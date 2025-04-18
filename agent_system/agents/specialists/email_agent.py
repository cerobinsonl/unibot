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
        
        # MailHog (local) configuration
        self.mailhog_server = os.getenv("SMTP_SERVER", "mailhog")
        self.mailhog_port   = int(os.getenv("SMTP_PORT", "1025"))

        # Mailtrap (online) configuration
        self.use_mailtrap  = os.getenv("USE_MAILTRAP", "false").lower() == "true"
        self.mailtrap_user = os.getenv("MAILTRAP_USERNAME", "")
        self.mailtrap_pass = os.getenv("MAILTRAP_PASSWORD", "")
        self.mailtrap_host = os.getenv("MAILTRAP_HOST", "sandbox.smtp.mailtrap.io")
        self.mailtrap_port = int(os.getenv("MAILTRAP_PORT", "2525"))

        # Email configuration
        self.from_email = os.getenv("FROM_EMAIL", "university-admin@example.edu")
        
        # Print configuration for debugging
        logger.info(f"Email Agent initialized with MailHog: {self.mailhog_server}:{self.mailhog_port}")
        
        # Create the email formatting prompt
        self.formatting_prompt = """
You are the Email Agent for a university administrative system.
Your role is to format and send emails to university stakeholders as **HTML**. Your task is to:

1. Format the **existing content** (provided in '{content}') into a clear and professional HTML email body.
2. Use `<p>` tags for paragraphs and `<br>` tags for single line breaks within paragraphs if necessary. Use `<ul>` and `<li>` for lists to structure the information.
3. **Do not add an additional greeting at the beginning of the email.** Assume the '{content}' already includes an appropriate salutation.
4. Ensure the tone matches the priority level and audience.
5. Add a professional university signature block at the very end of the email within `<p>` tags.

Format your response as a JSON object with these keys:
- formatted_subject: The revised subject line (plain text)
- formatted_content: The full **HTML** email body, incorporating the '{content}' with proper HTML structure and the university signature at the end.
- suggestions: Any suggestions for improving communication effectiveness

Example:
{{
    "formatted_subject": "Important Update: Final Exam Schedule Changes",
    "formatted_content": "<p>This is to inform you about important changes to the final exam schedule:</p><ul><li>New Date: ...</li><li>New Time: ...</li></ul>",
    "suggestions": ["Consider sending a follow-up reminder one week before exams", "Include a link to the full exam schedule"]
}}

Recipients: {recipients}
Subject: {subject}
Content: {content}
Priority: {priority}

Please format the **provided content** into a professional university **HTML** email, ensuring proper structure and adding only the university signature at the end. Do not add extra greetings or closings.
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
                formatted_content = f"{content}"
                suggestions = []
            
            # Determine if we should use mock sending or MailHog
            use_mailhog = os.getenv("USE_MAILHOG", "true").lower() == "true"
            
            if self.use_mailtrap:
                message_id, send_status = self._send_with_mailtrap(
                    recipients, formatted_subject, formatted_content, priority
                )
            else:
                message_id, send_status = self._send_with_mailhog(
                    recipients, formatted_subject, formatted_content, priority
                )

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
            msg.attach(MIMEText(content, 'html'))
            
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
    
    def _send_with_mailtrap(self, recipients, subject, content, priority):
        """
        Send email via Mailtrap SMTP (requires STARTTLS)
        """
        try:
            logger.info(f"Sending email via Mailtrap to: {recipients}")
            msg = MIMEMultipart()
            msg['From']    = self.from_email
            msg['To']      = ', '.join(recipients if isinstance(recipients, list) else [recipients])
            msg['Subject'] = subject

            # Priority headers
            if priority.lower() == "high":
                msg['X-Priority']        = '1'
                msg['X-MSMail-Priority'] = 'High'
                msg['Importance']        = 'High'

            msg.attach(MIMEText(content, 'html'))

            # Connect & secure
            server = smtplib.SMTP(self.mailtrap_host, self.mailtrap_port)
            server.starttls()  # <-- important for Mailtrap
            server.login(self.mailtrap_user, self.mailtrap_pass)
            server.sendmail(self.from_email, recipients, msg.as_string())
            server.quit()

            message_id = f"<{datetime.now():%Y%m%d%H%M%S}.{hash(content) % 10000}@mailtrap>"
            status = "success"

            self._log_email_details(recipients, subject, content, "MAILTRAP", message_id)
            return message_id, status

        except Exception as e:
            logger.error(f"Mailtrap error: {e}")
            return None, f"error: {e}"



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