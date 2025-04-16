from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

# Import configuration
from config import settings, AGENT_CONFIGS, get_llm

# Import tools
from tools.api_connectors import call_lms_api, call_sis_api, call_crm_api

# Configure logging
logger = logging.getLogger(__name__)

class IntegrationCoordinator:
    """
    Integration Coordinator manages connections to external systems like
    LMS (Learning Management System), SIS (Student Information System),
    and CRM (Customer Relationship Management) through API calls.
    """
    
    def __init__(self):
        """Initialize the Integration Coordinator"""
        # Create the LLM using the helper function
        self.llm = get_llm("integration_coordinator")
        
        # Create the task planning prompt
        self.planning_prompt = """
You are the Integration Coordinator for a university administrative system.
Your job is to manage connections to external systems like the LMS, SIS, and CRM.

You need to create a plan for retrieving data from external university systems. Determine which system and endpoint is needed:

1. LMS (Learning Management System) - For course content, assignments, grades, discussions
2. SIS (Student Information System) - For enrollment, transcripts, degree progress, financial aid
3. CRM (Customer Relationship Management) - For prospective student tracking, alumni relations, donations

Format your response as a JSON object with these keys:
- system: "lms", "sis", or "crm"
- endpoint: Specific API endpoint or data category to access
- parameters: Parameters needed for the API call
- purpose: Why this data is being requested

Example for LMS access:
{
  "system": "lms",
  "endpoint": "courses/assignments",
  "parameters": {
    "course_id": "BIO101",
    "term": "Fall2023"
  },
  "purpose": "Retrieve assignment submission rates for Biology 101"
}

Example for SIS access:
{
  "system": "sis",
  "endpoint": "student/enrollment",
  "parameters": {
    "department": "Computer Science",
    "year": "2023"
  },
  "purpose": "Get enrollment statistics for Computer Science department"
}

Important: Only request data that would be appropriate for university administrative staff to access.

User request: {user_input}
"""
        
        # Create the results synthesis prompt
        self.synthesis_prompt = """
You are the Integration Coordinator for a university administrative system.
Your job is to manage connections to external systems like the LMS, SIS, and CRM.

You are synthesizing the results from external system API calls to create a response for the user.

Review the integration request and the API results, then create a clear response that presents 
the retrieved data in a useful format for university administrators.

Your response should:
1. Mention which system provided the data
2. Summarize the key information retrieved
3. Present important metrics or statistics
4. Note any limitations or context for interpreting the data
5. Offer to get additional related information if needed

Be professional and concise, as appropriate for university administrative staff.

User request: {user_input}

Integration details: 
System: {system}
Endpoint: {endpoint}

API Results: {api_results}

Create a response synthesizing this information.
"""
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the integration request by coordinating with external systems
        
        Args:
            state: Current state of the conversation
            
        Returns:
            Updated state with integration results
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
                
                system_match = re.search(r'"system"\s*:\s*"([^"]+)"', planning_response)
                system = system_match.group(1) if system_match else "lms"
                
                endpoint_match = re.search(r'"endpoint"\s*:\s*"([^"]+)"', planning_response)
                endpoint = endpoint_match.group(1) if endpoint_match else "courses"
                
                # Try to extract parameters object
                params_match = re.search(r'"parameters"\s*:\s*(\{[^}]+\})', planning_response)
                if params_match:
                    try:
                        parameters = json.loads(params_match.group(1))
                    except:
                        parameters = {}
                else:
                    parameters = {}
                
                purpose_match = re.search(r'"purpose"\s*:\s*"([^"]+)"', planning_response)
                purpose = purpose_match.group(1) if purpose_match else "Retrieve data"
                
                plan = {
                    "system": system,
                    "endpoint": endpoint,
                    "parameters": parameters,
                    "purpose": purpose
                }
            
            # Add planning step to intermediate steps
            intermediate_steps.append({
                "agent": "integration",
                "action": "create_plan",
                "input": user_input,
                "output": plan,
                "timestamp": self._get_timestamp()
            })
            
            # Step 2: Execute the API call to the appropriate system
            api_result = None
            
            if plan["system"] == "lms":
                api_result = call_lms_api(plan["endpoint"], plan["parameters"])
                
                # Add LMS API call to intermediate steps
                intermediate_steps.append({
                    "agent": "integration",
                    "action": "call_lms_api",
                    "input": {
                        "endpoint": plan["endpoint"],
                        "parameters": plan["parameters"]
                    },
                    "output": api_result,
                    "timestamp": self._get_timestamp()
                })
                
            elif plan["system"] == "sis":
                api_result = call_sis_api(plan["endpoint"], plan["parameters"])
                
                # Add SIS API call to intermediate steps
                intermediate_steps.append({
                    "agent": "integration",
                    "action": "call_sis_api",
                    "input": {
                        "endpoint": plan["endpoint"],
                        "parameters": plan["parameters"]
                    },
                    "output": api_result,
                    "timestamp": self._get_timestamp()
                })
                
            elif plan["system"] == "crm":
                api_result = call_crm_api(plan["endpoint"], plan["parameters"])
                
                # Add CRM API call to intermediate steps
                intermediate_steps.append({
                    "agent": "integration",
                    "action": "call_crm_api",
                    "input": {
                        "endpoint": plan["endpoint"],
                        "parameters": plan["parameters"]
                    },
                    "output": api_result,
                    "timestamp": self._get_timestamp()
                })
            
            # Step 3: Synthesize results
            synthesis_input = {
                "user_input": user_input,
                "system": plan["system"].upper(),  # Make it uppercase for readability
                "endpoint": plan["endpoint"],
                "api_results": json.dumps(api_result, indent=2)
            }
            
            formatted_prompt = self.synthesis_prompt.format(**synthesis_input)
            response = self.llm.invoke(formatted_prompt).content
            
            # Update state
            state["response"] = response
            state["intermediate_steps"] = intermediate_steps
            state["current_agent"] = "integration"
            
            return state
            
        except Exception as e:
            logger.error(f"Error in Integration Coordinator: {e}", exc_info=True)
            error_response = f"I encountered an error while retrieving data from external systems: {str(e)}. Please try rephrasing your request or contact support if the issue persists."
            
            # Update state with error
            state["response"] = error_response
            state["current_agent"] = "integration"
            
            return state
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string"""
        return datetime.now().isoformat()