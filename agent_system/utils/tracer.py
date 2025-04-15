import logging
import json
from typing import Dict, Any, List
from datetime import datetime
import os

class AgentTracer:
    """
    Utility class for tracing agent communication in the LangGraph workflow.
    Records detailed information about agent activities and messages.
    """
    
    def __init__(self, trace_dir="./agent_traces"):
        """
        Initialize the tracer with the directory for storing traces
        
        Args:
            trace_dir: Directory to store trace files
        """
        self.logger = logging.getLogger("agent_tracer")
        self.trace_dir = trace_dir
        
        # Create trace directory if it doesn't exist
        os.makedirs(trace_dir, exist_ok=True)
        
        # Current trace data
        self.current_trace = {
            "start_time": datetime.now().isoformat(),
            "agents": {},
            "messages": [],
            "visualization_created": False
        }
        
        # Set up a file handler for this trace session
        self.trace_file = os.path.join(
            self.trace_dir, 
            f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    
    def record_agent_activity(self, agent_name: str, action: str, input_data: Any, output_data: Any):
        """
        Record an agent activity in the trace
        
        Args:
            agent_name: Name of the agent
            action: Action being performed
            input_data: Input data to the agent
            output_data: Output data from the agent
        """
        # Initialize agent info if not already present
        if agent_name not in self.current_trace["agents"]:
            self.current_trace["agents"][agent_name] = {
                "actions": [],
                "first_seen": datetime.now().isoformat()
            }
        
        # Record the action
        action_record = {
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "input": self._prepare_for_serialization(input_data),
            "output": self._prepare_for_serialization(output_data)
        }
        
        # Add to agent's actions
        self.current_trace["agents"][agent_name]["actions"].append(action_record)
        
        # Add to the chronological message list
        self.current_trace["messages"].append({
            "agent": agent_name,
            "action": action,
            "timestamp": action_record["timestamp"]
        })
        
        # Check for visualization
        if agent_name == "visualization_agent" and action == "create_visualization":
            self.current_trace["visualization_created"] = True
            
            # Record if the visualization data is present
            if isinstance(output_data, dict) and "image_data" in output_data:
                self.current_trace["visualization_data_present"] = True
                self.current_trace["visualization_type"] = output_data.get("chart_type", "unknown")
            else:
                self.current_trace["visualization_data_present"] = False
        
        # Log the activity
        self.logger.info(f"{agent_name} - {action}")
        
        # Save the updated trace
        self._save_trace()
    
    def record_state_update(self, state: Dict[str, Any]):
        """
        Record a state update in the trace
        
        Args:
            state: Current state of the conversation
        """
        # Record the state update
        state_record = {
            "timestamp": datetime.now().isoformat(),
            "user_input": state.get("user_input", ""),
            "current_agent": state.get("current_agent", ""),
            "has_response": "response" in state and state["response"] is not None,
            "has_visualization": "visualization" in state and state["visualization"] is not None,
            "step_count": len(state.get("intermediate_steps", []))
        }
        
        # Add to the trace
        if "state_updates" not in self.current_trace:
            self.current_trace["state_updates"] = []
        
        self.current_trace["state_updates"].append(state_record)
        
        # Log the state update
        self.logger.info(f"State update: agent={state_record['current_agent']}, " +
                         f"has_visualization={state_record['has_visualization']}")
        
        # Save the updated trace
        self._save_trace()
    
    def complete_trace(self, final_state: Dict[str, Any]):
        """
        Complete the trace with the final state
        
        Args:
            final_state: Final state of the conversation
        """
        # Record the final state
        self.current_trace["end_time"] = datetime.now().isoformat()
        self.current_trace["final_state"] = {
            "has_response": "response" in final_state and final_state["response"] is not None,
            "has_visualization": "visualization" in final_state and final_state["visualization"] is not None,
            "visualization_in_api_response": None  # To be filled by API layer
        }
        
        # Record visualization details if present
        if "visualization" in final_state and final_state["visualization"]:
            viz = final_state["visualization"]
            self.current_trace["final_state"]["visualization_details"] = {
                "chart_type": viz.get("chart_type", "unknown"),
                "has_image_data": "image_data" in viz and viz["image_data"] is not None,
                "image_type": viz.get("image_type", "unknown")
            }
        
        # Save the completed trace
        self._save_trace()
        
        # Log completion
        self.logger.info(f"Trace completed and saved to {self.trace_file}")
        
        return self.trace_file
    
    def _prepare_for_serialization(self, data: Any) -> Any:
        """
        Prepare data for JSON serialization, handling special cases
        
        Args:
            data: Data to prepare
            
        Returns:
            Serializable version of the data
        """
        if data is None:
            return None
            
        if isinstance(data, dict):
            # Handle dictionaries recursively
            result = {}
            for k, v in data.items():
                # Skip image data to avoid huge trace files
                if k == "image_data":
                    result[k] = f"[BINARY DATA: {len(str(v))} bytes]"
                else:
                    result[k] = self._prepare_for_serialization(v)
            return result
            
        if isinstance(data, list):
            # Handle lists recursively
            if len(data) > 10:
                # Truncate long lists
                return [self._prepare_for_serialization(x) for x in data[:10]] + ["..."]
            return [self._prepare_for_serialization(x) for x in data]
            
        if isinstance(data, (str, int, float, bool)):
            # Handle primitive types directly
            return data
            
        # For other types, convert to string
        return str(data)
    
    def _save_trace(self):
        """Save the current trace to the trace file"""
        with open(self.trace_file, 'w') as f:
            json.dump(self.current_trace, f, indent=2)

# Global instance that can be imported and used throughout the system
tracer = AgentTracer()

# Example usage:
# tracer.record_agent_activity("director", "analyze_intent", "Show student data", "ROUTE_TO_DATA_ANALYSIS")
# tracer.record_state_update(state)
# tracer.complete_trace(final_state)