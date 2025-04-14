from typing import Dict, List, Any, Optional, Callable
from langchain_core.language_models import BaseChatModel
import re
import json
import logging
from datetime import datetime

# Import configuration
from config import settings, AGENT_CONFIGS, get_llm

# Configure logging
logger = logging.getLogger(__name__)

class DirectorAgent:
    """
    Director Agent is responsible for understanding user intent,
    routing requests to appropriate coordinators, and presenting
    final responses to the user.
    """
    
    def __init__(self):
        """Initialize the Director Agent"""
        # Get the LLM using the helper function
        self.llm = get_llm("director")
        
        # Create the prompt template for initial user request processing
        self.intent_prompt = """
You are the Director Agent in a university administrative system. Your role is to understand user requests, coordinate with specialized teams, and present final responses to university staff.

University staff will ask you questions about student data, request data analysis, ask you to send messages, input data into databases, or extract information from university systems.

You must categorize each user request into one of these categories:
1. DATA_ANALYSIS - For data retrieval, analysis, and visualization
2. COMMUNICATION - For sending emails or messages
3. DATA_MANAGEMENT - For inputting or modifying database data
4. INTEGRATION - For retrieving data from external systems
5. CLARIFICATION - When you need more information from the user

When responding, include EXACTLY ONE of these routing tags at the beginning of your response:
- ROUTE_TO_DATA_ANALYSIS
- ROUTE_TO_COMMUNICATION
- ROUTE_TO_DATA_MANAGEMENT
- ROUTE_TO_INTEGRATION
- FINAL_RESPONSE (if you can provide a direct answer without calling other agents)

Example tags usage:
ROUTE_TO_DATA_ANALYSIS
I'll help you analyze the student enrollment data by department...

FINAL_RESPONSE
Here's the information you requested about our office hours...

User request: {user_input}
"""
        
        # Create the prompt template for final response synthesis
        self.synthesis_prompt = """
You are the Director Agent in a university administrative system. You are now synthesizing results from coordinator agents to provide a final response to the university staff member.

Review the full conversation history and the coordinator's response, then create a clear, helpful final response.

Your response should:
1. Be friendly, professional and concise
2. Emphasize the key insights or actions taken
3. Avoid technical jargon unless necessary
4. Include reference to any visualizations if they were created
5. Ask if the user needs anything else

Include FINAL_RESPONSE at the beginning of your message so the system knows this is the final answer.

User request: {user_input}

Conversation history: {history}

Coordinator response: {coordinator_response}

Please synthesize this information into a final response for the university staff member.
"""
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and determine next steps
        
        Args:
            state: Current state of the conversation
            
        Returns:
            Updated state with director's response
        """
        # Extract information from state
        user_input = state.get("user_input", "")
        history = state.get("history", [])
        current_agent = state.get("current_agent")
        intermediate_steps = state.get("intermediate_steps", [])
        
        # If we're coming from a coordinator, synthesize the final response
        if current_agent and current_agent != "director":
            # Get the most recent intermediate step from the coordinator
            coordinator_responses = [
                step["output"] for step in intermediate_steps 
                if step["agent"] == current_agent and step["output"] is not None
            ]
            
            if coordinator_responses:
                coordinator_response = coordinator_responses[-1]
                
                # Convert to string if it's a dictionary
                if isinstance(coordinator_response, dict):
                    coordinator_response = json.dumps(coordinator_response)
                
                # Synthesize the final response
                formatted_history = self._format_history_for_prompt(history)
                
                # Format the prompt with the required values
                formatted_prompt = self.synthesis_prompt.format(
                    user_input=user_input,
                    history=formatted_history,
                    coordinator_response=coordinator_response
                )
                
                # Invoke the LLM with the formatted prompt
                response = self.llm.invoke(formatted_prompt).content
                
                # Update state
                state["response"] = response
                state["current_agent"] = "director"
                
                return state
        
        # Initial processing of user request
        try:
            # Format the prompt with the user input
            formatted_prompt = self.intent_prompt.format(user_input=user_input)
            
            # Get director's analysis of the user request
            response = self.llm.invoke(formatted_prompt).content
            
            # Update state
            state["response"] = response
            state["current_agent"] = "director"
            
            # Add this step to intermediate steps
            intermediate_steps.append({
                "agent": "director",
                "action": "analyze_intent",
                "input": user_input,
                "output": response,
                "timestamp": self._get_timestamp()
            })
            
            state["intermediate_steps"] = intermediate_steps
            
            return state
            
        except Exception as e:
            logger.error(f"Error in Director Agent: {e}")
            error_response = f"FINAL_RESPONSE\nI apologize, but I encountered an error processing your request. Please try again or contact support if the issue persists."
            state["response"] = error_response
            return state
    
    def _format_history_for_prompt(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for the prompt"""
        if not history:
            return "No previous conversation."
        
        formatted = []
        for message in history:
            role = "User" if message["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {message['content']}")
        
        return "\n".join(formatted)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string"""
        return datetime.now().isoformat()