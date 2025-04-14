from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

# Import configuration
from config import settings, AGENT_CONFIGS, get_llm

# Import specialists
from agents.specialists.sql_agent import SQLAgent
from agents.specialists.analysis_agent import AnalysisAgent
from agents.specialists.visualization_agent import VisualizationAgent

# Configure logging
logger = logging.getLogger(__name__)

class DataAnalysisCoordinator:
    """
    Data Analysis Coordinator manages data retrieval, analysis, and visualization
    by delegating to specialized agents and orchestrating their work.
    """
    
    def __init__(self):
        """Initialize the Data Analysis Coordinator"""
        # Create the LLM using the helper function
        self.llm = get_llm("data_analysis_coordinator")
        
        # Initialize specialist agents
        self.sql_agent = SQLAgent()
        self.analysis_agent = AnalysisAgent()
        self.visualization_agent = VisualizationAgent()
        
        # Create the task planning prompt
        self.planning_prompt = """
You are the Data Analysis Coordinator for a university administrative system.
Your job is to coordinate data retrieval, analysis, and visualization tasks.

You need to create a plan for handling this data analysis request. Break it down into specific tasks for:

1. SQL Query Agent - What data needs to be retrieved from the database?
2. Analysis Agent - What analysis should be performed on this data?
3. Visualization Agent - What visualization would best represent this data?

Format your response as a JSON object with these keys:
- sql_task: Detailed description of what the SQL Agent should do
- analysis_task: Detailed description of what the Analysis Agent should do  
- visualization_task: Detailed description of what the Visualization Agent should do
- needs_visualization: true/false whether this request requires a visualization

Example:
{
  "sql_task": "Retrieve student enrollment counts by department for the last academic year",
  "analysis_task": "Calculate department growth rates compared to previous year and identify top 5 growing departments",
  "visualization_task": "Create a bar chart showing enrollment by department with growth indicators",
  "needs_visualization": true
}

Important: Make your descriptions specific and detailed so each specialized agent knows exactly what to do.

User request: {user_input}
"""
        
        # Create the results synthesis prompt
        self.synthesis_prompt = """
You are the Data Analysis Coordinator for a university administrative system.
Your job is to coordinate data retrieval, analysis, and visualization tasks.

You are synthesizing the results from specialist agents to create a comprehensive response.

Review the SQL query results, analysis results, and visualization (if available), then create 
a detailed response that explains the findings clearly for university administrators.

Your response should:
1. Summarize the key insights from the data
2. Explain any important patterns or trends
3. Reference the visualization if one was created
4. Provide context to help interpret the results
5. Be written in clear, non-technical language suitable for university staff

If there were any issues or limitations with the data, mention them briefly.

User request: {user_input}

SQL Query Results: {sql_results}

Analysis Results: {analysis_results}

Visualization: {has_visualization}

Create a comprehensive response synthesizing all this information.
"""
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the data analysis request by coordinating specialist agents
        
        Args:
            state: Current state of the conversation
            
        Returns:
            Updated state with analysis results
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
                # If the response isn't valid JSON, extract what we can using regex
                import re
                
                sql_task = re.search(r'"sql_task"\s*:\s*"([^"]+)"', planning_response)
                sql_task = sql_task.group(1) if sql_task else "Retrieve relevant data from the database"
                
                analysis_task = re.search(r'"analysis_task"\s*:\s*"([^"]+)"', planning_response)
                analysis_task = analysis_task.group(1) if analysis_task else "Analyze the retrieved data"
                
                visualization_task = re.search(r'"visualization_task"\s*:\s*"([^"]+)"', planning_response)
                visualization_task = visualization_task.group(1) if visualization_task else "Create a visualization of the data"
                
                needs_visualization = "true" in planning_response.lower()
                
                plan = {
                    "sql_task": sql_task,
                    "analysis_task": analysis_task,
                    "visualization_task": visualization_task,
                    "needs_visualization": needs_visualization
                }
            
            # Add planning step to intermediate steps
            intermediate_steps.append({
                "agent": "data_analysis",
                "action": "create_plan",
                "input": user_input,
                "output": plan,
                "timestamp": self._get_timestamp()
            })
            
            # Step 2: Execute SQL query
            sql_result = self.sql_agent(plan["sql_task"])
            
            # Add SQL step to intermediate steps
            intermediate_steps.append({
                "agent": "sql_agent",
                "action": "execute_query",
                "input": plan["sql_task"],
                "output": sql_result,
                "timestamp": self._get_timestamp()
            })
            
            # Step 3: Perform analysis
            analysis_result = self.analysis_agent({
                "task": plan["analysis_task"],
                "data": sql_result["results"],
                "column_names": sql_result["column_names"]
            })
            
            # Add analysis step to intermediate steps
            intermediate_steps.append({
                "agent": "analysis_agent",
                "action": "analyze_data",
                "input": {
                    "task": plan["analysis_task"],
                    "data": "Data from SQL query"  # Don't log full data for brevity
                },
                "output": analysis_result,
                "timestamp": self._get_timestamp()
            })
            
            # Step 4: Create visualization if needed
            visualization_result = None
            if plan["needs_visualization"]:
                visualization_result = self.visualization_agent({
                    "task": plan["visualization_task"],
                    "data": sql_result["results"],
                    "column_names": sql_result["column_names"],
                    "analysis": analysis_result
                })
                
                # Add visualization step to intermediate steps
                intermediate_steps.append({
                    "agent": "visualization_agent",
                    "action": "create_visualization",
                    "input": {
                        "task": plan["visualization_task"],
                        "data": "Data from SQL query"  # Don't log full data for brevity
                    },
                    "output": "Visualization created",  # Don't log binary data
                    "timestamp": self._get_timestamp()
                })
                
                # Add visualization to state
                state["visualization"] = visualization_result
            
            # Step 5: Synthesize results
            synthesis_input = {
                "user_input": user_input,
                "sql_results": self._format_sql_results(sql_result),
                "analysis_results": analysis_result["summary"],
                "has_visualization": "Yes, a visualization was created and attached." if visualization_result else "No visualization was created."
            }
            
            formatted_prompt = self.synthesis_prompt.format(**synthesis_input)
            response = self.llm.invoke(formatted_prompt).content
            
            # Update state
            state["response"] = response
            state["intermediate_steps"] = intermediate_steps
            state["current_agent"] = "data_analysis"
            
            return state
            
        except Exception as e:
            logger.error(f"Error in Data Analysis Coordinator: {e}", exc_info=True)
            error_response = f"I encountered an error while analyzing the data: {str(e)}. Please try rephrasing your request or contact support if the issue persists."
            
            # Update state
            state["response"] = error_response
            state["current_agent"] = "data_analysis"
            
            return state
    
    def _format_sql_results(self, sql_result: Dict[str, Any]) -> str:
        """Format SQL results for readability in the prompt"""
        if not sql_result or not sql_result.get("results"):
            return "No data retrieved."
        
        results = sql_result["results"]
        column_names = sql_result.get("column_names", [])
        
        # Limit to a reasonable number of rows for the prompt
        max_rows = 10
        sample_results = results[:max_rows]
        
        # Format as text table
        formatted = "SQL Query Results:\n"
        formatted += f"Retrieved {len(results)} rows with columns: {', '.join(column_names)}\n\n"
        
        if len(sample_results) > 0:
            formatted += "Sample data:\n"
            import tabulate
            try:
                # Try to format as a table using tabulate
                formatted += tabulate.tabulate(
                    [[row.get(col) for col in column_names] for row in sample_results],
                    headers=column_names,
                    tablefmt="pipe"
                )
            except:
                # Fall back to simple formatting
                for i, row in enumerate(sample_results):
                    formatted += f"Row {i+1}: {str(row)}\n"
        
        return formatted
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string"""
        return datetime.now().isoformat()