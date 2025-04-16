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
5. Use precise numbers and specific information from the retrieved data when available
6. DO NOT use placeholders like "[Value from Visualization]" - either include the actual value or rephrase without a placeholder
7. DO NOT include phrases like "Let me know if you need anything else" or "Is there anything else you'd like to know?"
8. DO NOT start your response with "FINAL_RESPONSE" or any other prefix tag

Provide a direct answer to the university staff member's query without unnecessary formalities.

User request: {user_input}

Conversation history: {history}

Coordinator response: {coordinator_response}

Retrieved data: {retrieved_data}

Has visualization: {has_visualization}

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
        # Make a copy of the state to avoid modifying shared state
        working_state = {}
        for key in ['user_input', 'session_id', 'history', 'current_agent', 
                    'response', 'intermediate_steps', 'visualization', 'stream']:
            if key in state:
                working_state[key] = state[key]
        
        # Extract information from state
        user_input = working_state.get("user_input", "")
        history = working_state.get("history", [])
        current_agent = working_state.get("current_agent")
        intermediate_steps = working_state.get("intermediate_steps", [])
        visualization = working_state.get("visualization", None)
        
        # Log the current state for debugging agent communication
        logger.info(f"===== DIRECTOR AGENT STATE =====")
        logger.info(f"User input: {user_input}")
        logger.info(f"Current agent: {current_agent}")
        logger.info(f"Has visualization: {visualization is not None}")
        logger.info(f"Intermediate steps count: {len(intermediate_steps) if intermediate_steps else 0}")
        
        # Initialize intermediate_steps if it's None
        if intermediate_steps is None:
            intermediate_steps = []
            working_state["intermediate_steps"] = intermediate_steps
        
        # If we're coming from a coordinator, synthesize the final response
        if current_agent and current_agent != "director":
            # Log the flow
            logger.info(f"===== SYNTHESIZING RESPONSE FROM {current_agent.upper()} =====")
            
            # Get the most recent intermediate step from the coordinator
            coordinator_responses = [
                step["output"] for step in intermediate_steps 
                if step["agent"] == current_agent and step["output"] is not None
            ]
            
            coordinator_response = ""
            if coordinator_responses:
                last_response = coordinator_responses[-1]
                
                # Convert to string if it's a dictionary
                if isinstance(last_response, dict):
                    coordinator_response = json.dumps(last_response)
                else:
                    coordinator_response = str(last_response)
                
                # Log the coordinator's response
                logger.info(f"Coordinator response: {coordinator_response[:500]}...")
            
            # Look for retrieved data (especially from SQL queries)
            retrieved_data = "No specific data retrieved."
            for step in intermediate_steps:
                if step["agent"] == "sql_agent" and "output" in step and step["output"] is not None:
                    sql_result = step["output"]
                    if isinstance(sql_result, dict) and "results" in sql_result:
                        if sql_result["results"]:
                            # Get the first few results to include in the prompt
                            retrieved_data = json.dumps(sql_result["results"][:3], indent=2)
                            # Log the retrieved data
                            logger.info(f"Retrieved data: {retrieved_data}")
                        else:
                            retrieved_data = "Query executed successfully but returned no results."
            
            # Check if there's a visualization
            has_visualization = "Yes" if visualization else "No"
            logger.info(f"Has visualization in state: {has_visualization}")
            
            # Log detailed intermediate steps for agent communication tracing
            logger.info("===== INTERMEDIATE STEPS =====")
            for i, step in enumerate(intermediate_steps):
                agent = step.get("agent", "unknown")
                action = step.get("action", "unknown")
                timestamp = step.get("timestamp", "")
                
                # Format the input and output for readability
                input_str = str(step.get("input", ""))
                if len(input_str) > 200:
                    input_str = input_str[:200] + "..."
                
                output = step.get("output", None)
                output_str = "None"
                if output:
                    if isinstance(output, dict):
                        if "results" in output:
                            output_str = f"Results with {len(output['results'])} items"
                        else:
                            output_keys = list(output.keys())
                            output_str = f"Dict with keys: {output_keys}"
                    else:
                        output_str = str(output)
                        if len(output_str) > 200:
                            output_str = output_str[:200] + "..."
                
                logger.info(f"Step {i+1}: {agent} -> {action}")
                logger.info(f"  Input: {input_str}")
                logger.info(f"  Output: {output_str}")
                logger.info(f"  Time: {timestamp}")
            
            # Synthesize the final response
            formatted_history = self._format_history_for_prompt(history)
            
            # Format the prompt with the required values
            formatted_prompt = self.synthesis_prompt.format(
                user_input=user_input,
                history=formatted_history,
                coordinator_response=coordinator_response,
                retrieved_data=retrieved_data,
                has_visualization=has_visualization
            )
            
            # Log the synthesis prompt
            logger.info("===== SYNTHESIS PROMPT =====")
            logger.info(formatted_prompt[:500] + "...")
            
            # Invoke the LLM with the formatted prompt
            response = self.llm.invoke(formatted_prompt).content
            
            # Log the synthesis result
            logger.info(f"===== SYNTHESIZED RESPONSE =====")
            logger.info(f"{response[:500]}...")
            
            # Update state
            working_state["response"] = response
            working_state["current_agent"] = "director"
            
            # Copy back to original state
            for key, value in working_state.items():
                state[key] = value
                
            # Set a flag to indicate this is a final response
            # This is added separately from the working_state to avoid validation errors
            state["is_final_response"] = True
            
            return state
        
        # Initial processing of user request
        try:
            # Log that we're processing the initial request
            logger.info(f"===== PROCESSING INITIAL REQUEST =====")
            
            # Format the prompt with the user input
            formatted_prompt = self.intent_prompt.format(user_input=user_input)
            
            # Log the intent prompt
            logger.info("===== INTENT PROMPT =====")
            logger.info(formatted_prompt[:500] + "...")
            
            # Get director's analysis of the user request
            response = self.llm.invoke(formatted_prompt).content
            
            # Log the intent response
            logger.info("===== INTENT RESPONSE =====")
            logger.info(response[:500] + "...")
            
            # Check if visualization is explicitly requested
            visualization_requested = any(keyword in user_input.lower() for keyword in 
                ['chart', 'plot', 'graph', 'visualization', 'visualize', 'visualisation', 
                'histogram', 'bar chart', 'show me', 'display'])
            
            # Update state
            working_state["response"] = response
            working_state["current_agent"] = "director"
            
            # Add this step to intermediate steps
            intermediate_steps.append({
                "agent": "director",
                "action": "analyze_intent",
                "input": user_input,
                "output": response,
                "timestamp": self._get_timestamp()
            })
            
            working_state["intermediate_steps"] = intermediate_steps
            
            # Copy back to original state
            for key, value in working_state.items():
                state[key] = value
                
            # Add visualization_requested flag separately to avoid validation errors
            state["visualization_requested"] = visualization_requested
            
            # Log the routing decision
            if "ROUTE_TO_DATA_ANALYSIS" in response:
                logger.info("Routing to DATA_ANALYSIS")
            elif "ROUTE_TO_COMMUNICATION" in response:
                logger.info("Routing to COMMUNICATION")
            elif "ROUTE_TO_DATA_MANAGEMENT" in response:
                logger.info("Routing to DATA_MANAGEMENT")
            elif "ROUTE_TO_INTEGRATION" in response:
                logger.info("Routing to INTEGRATION")
            elif "FINAL_RESPONSE" in response:
                logger.info("Providing FINAL_RESPONSE directly")
            else:
                logger.info("No clear routing found in response")
            
            return state
                
        except Exception as e:
            logger.error(f"Error in Director Agent: {e}", exc_info=True)
            error_response = f"I apologize, but I encountered an error processing your request. Please try again or contact support if the issue persists."
            working_state["response"] = error_response
            
            # Copy back to original state
            for key, value in working_state.items():
                state[key] = value
                
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