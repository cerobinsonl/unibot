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
    
    # Add nodes to the graph
    workflow.add_node("director", director_agent)
    workflow.add_node("data_analysis", data_analysis_coordinator)
    workflow.add_node("communication", communication_coordinator)
    workflow.add_node("data_management", data_management_coordinator)
    workflow.add_node("integration", integration_coordinator)
    
    # Define the director's routing logic
    def route_request(state: GraphState) -> str:
        """
        Determine which coordinator should handle the request
        
        Args:
            state: Current conversation state
            
        Returns:
            Name of the next node to route to
        """
        current_agent = state.get("current_agent")
        
        # If a current agent is already assigned, return it
        if current_agent and current_agent != "director":
            return current_agent
            
        # Extract intent from director's response
        response = state.get("response", "")
        
        try:
            # Attempt to parse routing information from director response
            if "ROUTE_TO_DATA_ANALYSIS" in response:
                return "data_analysis"
            elif "ROUTE_TO_COMMUNICATION" in response:
                return "communication"
            elif "ROUTE_TO_DATA_MANAGEMENT" in response:
                return "data_management"
            elif "ROUTE_TO_INTEGRATION" in response:
                return "integration"
            elif "FINAL_RESPONSE" in response:
                return END
            else:
                # Default to data analysis for now as it's most common
                # In production, should log this as a potential issue
                logger.warning(f"No clear routing found in: {response[:100]}...")
                return "data_analysis"
        except Exception as e:
            logger.error(f"Error in routing: {e}")
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
    
    # Compile the graph
    return workflow.compile()