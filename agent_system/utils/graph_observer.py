import logging
from typing import Dict, Any, Optional
from langchain_core.messages import BaseMessage
import json

# Configure logging
logger = logging.getLogger(__name__)

class LangGraphObserver:
    """
    Observer for LangGraph execution that logs agent interactions
    """
    
    def __init__(self, log_file: Optional[str] = "langgraph_observations.jsonl"):
        """
        Initialize the observer
        
        Args:
            log_file: File to log observations to (None to disable file logging)
        """
        self.log_file = log_file
        self.observations = []
    
    def on_start(self, serialized: Dict[str, Any]) -> None:
        """Called when the graph execution starts"""
        logger.info("===== GRAPH EXECUTION STARTED =====")
        
        self._log_observation({
            "event": "start",
            "data": serialized
        })
    
    def on_end(self, serialized: Dict[str, Any]) -> None:
        """Called when the graph execution ends"""
        logger.info("===== GRAPH EXECUTION COMPLETED =====")
        
        # Log the final state keys
        if "state" in serialized:
            state_keys = list(serialized["state"].keys())
            logger.info(f"Final state keys: {state_keys}")
            
            # Check for visualization
            if "visualization" in serialized["state"]:
                viz = serialized["state"]["visualization"]
                if viz:
                    logger.info("Visualization present in final state")
                    if isinstance(viz, dict):
                        logger.info(f"Visualization type: {viz.get('chart_type', 'unknown')}")
        
        self._log_observation({
            "event": "end",
            "data": serialized
        })
    
    def on_node_start(self, serialized: Dict[str, Any]) -> None:
        """Called when a node starts execution"""
        node_name = serialized.get("node", {}).get("id", "unknown")
        logger.info(f"===== NODE STARTED: {node_name} =====")
        
        self._log_observation({
            "event": "node_start",
            "node": node_name,
            "data": serialized
        })
    
    def on_node_end(self, serialized: Dict[str, Any]) -> None:
        """Called when a node ends execution"""
        node_name = serialized.get("node", {}).get("id", "unknown")
        logger.info(f"===== NODE COMPLETED: {node_name} =====")
        
        # Check if the node is data_analysis and look for visualization
        if node_name == "data_analysis" and "state" in serialized:
            if "visualization" in serialized["state"]:
                viz = serialized["state"]["visualization"]
                if viz:
                    logger.info("Visualization created by data_analysis node")
        
        self._log_observation({
            "event": "node_end",
            "node": node_name,
            "data": serialized
        })
    
    def on_chain_start(self, serialized: Dict[str, Any]) -> None:
        """Called when a chain starts"""
        self._log_observation({
            "event": "chain_start",
            "data": serialized
        })
    
    def on_chain_end(self, serialized: Dict[str, Any]) -> None:
        """Called when a chain ends"""
        self._log_observation({
            "event": "chain_end",
            "data": serialized
        })
    
    def _log_observation(self, observation: Dict[str, Any]) -> None:
        """Log an observation to the file and memory"""
        self.observations.append(observation)
        
        if self.log_file:
            try:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(observation) + "\n")
            except Exception as e:
                logger.error(f"Error logging observation: {e}")