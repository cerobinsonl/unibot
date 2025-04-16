from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import logging
import json
from pydantic import BaseModel

# Import config
from config import settings, AGENT_CONFIGS, get_llm

# Import agents
from agents.director import DirectorAgent
from agents.coordinators.data_analysis import DataAnalysisCoordinator
from agents.coordinators.communication import CommunicationCoordinator
from agents.coordinators.data_management import DataManagementCoordinator
from agents.coordinators.integration import IntegrationCoordinator

# Import observer for monitoring
from utils.graph_observer import LangGraphObserver

# Import agent tracer
from utils.tracer import tracer

# Configure logging
logger = logging.getLogger(__name__)

# Define state for the agent graph
class AgentState(BaseModel):
    """State for the agent graph"""
    session_id: str
    workflow: Any = None
    history: List[Dict[str, str]] = []

# Define the agent graph input
class GraphState(TypedDict):
    """Input and state for the graph"""
    user_input: str
    session_id: str
    history: List[Dict[str, str]]
    current_agent: Optional[str]
    response: Optional[str]
    intermediate_steps: List[Dict[str, Any]]
    visualization: Optional[Dict[str, Any]]
    stream: Optional[bool]
    is_final_response: Optional[bool]  # Add this line
    visualization_requested: Optional[bool]  # Add this line

def create_workflow(streaming: bool = False) -> StateGraph:
    """
    Create the LangGraph workflow that orchestrates the agent hierarchy
    
    Args:
        streaming: Whether to enable streaming mode
        
    Returns:
        The compiled workflow graph
    """
    # Initialize agents
    director_agent = DirectorAgent()
    data_analysis_coordinator = DataAnalysisCoordinator()
    communication_coordinator = CommunicationCoordinator()
    data_management_coordinator = DataManagementCoordinator()
    integration_coordinator = IntegrationCoordinator()
    
    # Define the workflow graph
    workflow = StateGraph(GraphState)
    
    # Define custom data_analysis function that ensures visualization is preserved
    def data_analysis_with_preservation(state: GraphState) -> GraphState:
        """Run data analysis and ensure visualization is preserved"""
        # Log that we're running this custom function
        logger.info("Running data analysis with visualization preservation")
        
        # Record in tracer
        tracer.record_agent_activity("data_analysis", "start", state.get("user_input", ""), None)
        
        # Run the normal data analysis coordinator
        result_state = data_analysis_coordinator(state)
        
        # Ensure the state indicates we're in data_analysis mode
        result_state["current_agent"] = "data_analysis"
        
        # Log visualization status for debugging
        if "visualization" in result_state and result_state["visualization"] is not None:
            logger.info("Found visualization in data analysis result, preserving for final response")
            # Record visualization creation in tracer
            tracer.record_agent_activity("data_analysis", "create_visualization", 
                                         result_state.get("user_input", ""), 
                                         {"has_visualization": True})
        
        # Record state update in tracer
        tracer.record_state_update(result_state)
        
        # Record completion in tracer
        tracer.record_agent_activity("data_analysis", "complete", state.get("user_input", ""), result_state)
        
        return result_state
    
    # Custom wrapper for director agent
    def director_with_tracing(state: GraphState) -> GraphState:
        """Run director agent with tracing"""
        # Record in tracer
        tracer.record_agent_activity("director", "start", state.get("user_input", ""), None)
        
        # Run the director agent
        result_state = director_agent(state)
        
        # Record state update in tracer
        tracer.record_state_update(result_state)
        
        # Record completion in tracer
        tracer.record_agent_activity("director", "complete", state.get("user_input", ""), result_state)
        
        return result_state
    
    # Custom wrapper for communication coordinator
    def communication_with_tracing(state: GraphState) -> GraphState:
        """Run communication coordinator with tracing"""
        # Record in tracer
        tracer.record_agent_activity("communication", "start", state.get("user_input", ""), None)
        
        # Run the communication coordinator
        result_state = communication_coordinator(state)
        
        # Record state update in tracer
        tracer.record_state_update(result_state)
        
        # Record completion in tracer
        tracer.record_agent_activity("communication", "complete", state.get("user_input", ""), result_state)
        
        return result_state
    
    # Custom wrapper for data management coordinator
    def data_management_with_tracing(state: GraphState) -> GraphState:
        """Run data management coordinator with tracing"""
        # Record in tracer
        tracer.record_agent_activity("data_management", "start", state.get("user_input", ""), None)
        
        # Run the data management coordinator
        result_state = data_management_coordinator(state)
        
        # Record state update in tracer
        tracer.record_state_update(result_state)
        
        # Record completion in tracer
        tracer.record_agent_activity("data_management", "complete", state.get("user_input", ""), result_state)
        
        return result_state
    
    # Custom wrapper for integration coordinator
    def integration_with_tracing(state: GraphState) -> GraphState:
        """Run integration coordinator with tracing"""
        # Record in tracer
        tracer.record_agent_activity("integration", "start", state.get("user_input", ""), None)
        
        # Run the integration coordinator
        result_state = integration_coordinator(state)
        
        # Record state update in tracer
        tracer.record_state_update(result_state)
        
        # Record completion in tracer
        tracer.record_agent_activity("integration", "complete", state.get("user_input", ""), result_state)
        
        return result_state
    
    # Add nodes to the graph
    workflow.add_node("director", director_with_tracing)
    workflow.add_node("data_analysis", data_analysis_with_preservation)
    workflow.add_node("communication", communication_with_tracing)
    workflow.add_node("data_management", data_management_with_tracing)
    workflow.add_node("integration", integration_with_tracing)
    
    # Define the director's routing logic
    def route_request(state: GraphState) -> str:
        """
        Determine which coordinator should handle the request
        
        Args:
            state: Current state of the conversation
            
        Returns:
            Name of the next node to route to
        """
        # Copy only the allowed fields to avoid state validation errors
        allowed_fields = [
            'user_input', 'session_id', 'history', 'current_agent', 
            'response', 'intermediate_steps', 'visualization', 'stream'
        ]
        
        # Save any extra fields we want to preserve
        is_final = state.get("is_final_response", False)
        viz_requested = state.get("visualization_requested", False)
        
        # Extract rest of the information from state
        current_agent = state.get("current_agent")
        
        # Check if this is marked as a final response that shouldn't be routed again
        if is_final:
            logger.info("Final response detected, ending conversation")
            return END
        
        # If a current agent is already assigned, return it
        if current_agent and current_agent != "director":
            return current_agent
            
        # Extract intent from director's response
        response = state.get("response", "")
        
        try:
            # Attempt to parse routing information from director response
            route_result = None
            
            if "ROUTE_TO_DATA_ANALYSIS" in response:
                route_result = "data_analysis"
                # Add back our saved flags
                if viz_requested:
                    state["visualization_requested"] = viz_requested
            elif "ROUTE_TO_COMMUNICATION" in response:
                route_result = "communication"
            elif "ROUTE_TO_DATA_MANAGEMENT" in response:
                route_result = "data_management"
            elif "ROUTE_TO_INTEGRATION" in response:
                route_result = "integration"
            elif "FINAL_RESPONSE" in response:
                # For final responses, clean up the prefix and update the response
                cleaned_response = re.sub(r'^FINAL_RESPONSE\s*', '', response)
                state["response"] = cleaned_response
                route_result = END
            else:
                # Default to end if no clear routing is found
                logger.warning(f"No clear routing found in: {response[:100]}...")
                route_result = END
            
            # Record routing decision in tracer
            tracer.record_agent_activity("router", "route", response[:100], {"route_to": route_result})
            
            return route_result
            
        except Exception as e:
            logger.error(f"Error in routing: {e}")
            # Record error in tracer
            tracer.record_agent_activity("router", "error", response[:100], {"error": str(e)})
            # Default to END on errors
            return END
    
    # Define edges - the flow between agents
    # Start with the director
    workflow.set_entry_point("director")
    
    # Director routes to the appropriate coordinator
    workflow.add_conditional_edges(
        "director",
        route_request,
        {
            "data_analysis": "data_analysis",
            "communication": "communication",
            "data_management": "data_management",
            "integration": "integration",
            END: END
        }
    )
    
    # All coordinators return to the director for final processing
    workflow.add_edge("data_analysis", "director")
    workflow.add_edge("communication", "director")
    workflow.add_edge("data_management", "director")
    workflow.add_edge("integration", "director")
    
    # Create and add the observer for monitoring
    #observer = LangGraphObserver()
    compiled_graph = workflow.compile()
    
    # Add observer to graph
    #compiled_graph.add_observer(observer)
    
    return compiled_graph