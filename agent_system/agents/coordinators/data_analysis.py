from typing import Dict, List, Any, Optional
import json
import logging
import tabulate
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

IMPORTANT GUIDELINES:
- Do not make assumptions about database structure - SQL Agent will determine the appropriate tables and columns
- Avoid hardcoding column names or table names in your instructions
- For SQL tasks, focus on describing what data is needed, not how to query it
- For analysis tasks, describe the insights needed, not specific column operations
- For visualization tasks, describe what should be visualized, not how to code it
- Let each specialist agent determine how to implement their tasks based on the actual data structure

Format your response as a JSON object with these keys:
- sql_task: Detailed description of what the SQL Agent should do
- analysis_task: Detailed description of what the Analysis Agent should do  
- visualization_task: Detailed description of what the Visualization Agent should do
- needs_visualization: true/false whether this request requires a visualization

Example:
{{
  "sql_task": "Retrieve student enrollment counts broken down by academic department for the current academic year",
  "analysis_task": "Analyze the enrollment distribution across departments. Calculate the percentage of total enrollment for each department and identify departments with significant enrollment changes.",
  "visualization_task": "Create a visualization showing the distribution of enrollment across departments. Use an appropriate chart type that makes it easy to compare department sizes.",
  "needs_visualization": true
}}

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
2. Use specific numbers and statistics from the query results
3. Explain any important patterns or trends
4. Reference the visualization if one was created
5. Provide context to help interpret the results
6. Be written in clear, non-technical language suitable for university staff

If there were any issues or limitations with the data, mention them briefly.

User request: {user_input}

SQL Query: {sql_query}

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
            
            # Get the visualization_requested flag if present, but handle case where it isn't
            visualization_requested = False
            if "visualization_requested" in state:
                visualization_requested = state["visualization_requested"]
            else:
                # If the flag isn't present, check keywords in the user input
                visualization_requested = any(keyword in user_input.lower() for keyword in 
                    ['chart', 'plot', 'graph', 'visualization', 'visualize', 'visualisation', 
                    'histogram', 'bar chart', 'show me', 'display'])
            
            logger.info(f"Data Analysis Coordinator processing: '{user_input}'")
            logger.info(f"Visualization explicitly requested: {visualization_requested}")
            
            # Step 1: Create a plan for handling the request
            formatted_prompt = self.planning_prompt.format(user_input=user_input)
            planning_response = self.llm.invoke(formatted_prompt).content
            
            # Log the planning response
            logger.debug(f"Planning response: {planning_response}")
            
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
            
            # Log the plan
            logger.info(f"Plan: {json.dumps(plan)}")
            
            # Add planning step to intermediate steps
            intermediate_steps.append({
                "agent": "data_analysis",
                "action": "create_plan",
                "input": user_input,
                "output": plan,
                "timestamp": self._get_timestamp()
            })
            
            # Step 2: Execute SQL query
            logger.info(f"Executing SQL query: {plan['sql_task']}")
            sql_result = self.sql_agent(plan["sql_task"])
            
            # Log the query and results
            if "query" in sql_result:
                logger.info(f"SQL query: {sql_result['query']}")
            
            logger.info(f"SQL result row count: {sql_result.get('row_count', 0)}")
            
            # Add SQL step to intermediate steps
            intermediate_steps.append({
                "agent": "sql_agent",
                "action": "execute_query",
                "input": plan["sql_task"],
                "output": sql_result,
                "timestamp": self._get_timestamp()
            })
            
            # Check if there was an error with the SQL query
            if sql_result.get("is_error", False):
                logger.warning(f"SQL query error: {sql_result.get('error', 'Unknown error')}")
                return self._handle_sql_error(state, user_input, sql_result, intermediate_steps)
            
            # Step 3: Perform analysis
            logger.info(f"Performing analysis: {plan['analysis_task']}")
            analysis_result = self.analysis_agent({
                "task": plan["analysis_task"],
                "data": sql_result["results"],
                "column_names": sql_result["column_names"],
                "row_count": sql_result["row_count"]  # Add row count information
            })
            
            # Log the analysis summary
            logger.info(f"Analysis summary: {analysis_result.get('summary', '')[:100]}...")
            
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
            
            # Step 4: Create visualization if needed and explicitly requested
            visualization_result = None
            should_visualize = plan["needs_visualization"] and (visualization_requested or 
                                                          any(keyword in user_input.lower() for keyword in 
                                                          ['chart', 'plot', 'graph', 'visualization', 'visualize', 
                                                          'visualisation', 'histogram', 'bar chart', 'show me', 'display',
                                                          'distribution', 'breakdown', 'trend']))
            
            if should_visualize:
                logger.info(f"Creating visualization: {plan['visualization_task']}")
                
                visualization_result = self.visualization_agent({
                    "task": plan["visualization_task"],
                    "data": sql_result["results"],
                    "column_names": sql_result["column_names"],
                    "analysis": analysis_result,
                    "query": sql_result["query"]  # Add the original query for context
                })
                
                # Log visualization creation result
                logger.info(f"Visualization created: {visualization_result.get('chart_type', 'unknown type')}")
                
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
                
                # Ensure visualization is properly added to state
                if "visualization" not in state or state["visualization"] is None:
                    state["visualization"] = visualization_result
                
                # Double check that image_data exists
                if "image_data" not in visualization_result or not visualization_result["image_data"]:
                    logger.error("Visualization result missing image_data")
                else:
                    logger.info(f"Visualization added to state with image_data length: {len(visualization_result['image_data'])}")
            
            # Step 5: Synthesize results
            synthesis_input = {
                "user_input": user_input,
                "sql_query": sql_result.get("query", ""),
                "sql_results": self._format_sql_results(sql_result),
                "analysis_results": analysis_result["summary"],
                "has_visualization": "Yes, a visualization was created and attached." if visualization_result else "No visualization was created."
            }
            
            formatted_prompt = self.synthesis_prompt.format(**synthesis_input)
            logger.info("Generating final synthesis response")
            response = self.llm.invoke(formatted_prompt).content
            
            # Log the synthesis response
            logger.info(f"Synthesis response: {response[:100]}...")
            
            # Update state
            state["response"] = response
            state["intermediate_steps"] = intermediate_steps
            state["current_agent"] = "data_analysis"
            
            # Double-check visualization is in state if we created one
            if visualization_result and "visualization" not in state:
                logger.warning("Visualization was created but not found in state, adding it now")
                state["visualization"] = visualization_result
            
            # Log final state
            logger.info(f"Final state has visualization: {'visualization' in state and state['visualization'] is not None}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in Data Analysis Coordinator: {e}", exc_info=True)
            error_response = f"I encountered an error while analyzing the data: {str(e)}. Please try rephrasing your request or contact support if the issue persists."
            
            # Update state
            state["response"] = error_response
            state["current_agent"] = "data_analysis"
            
            return state
    
    def _handle_sql_error(self, state: Dict[str, Any], user_input: str, sql_result: Dict[str, Any], intermediate_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle SQL query errors with a graceful response"""
        
        error_message = sql_result.get("error", "Unknown database error")
        query_attempted = sql_result.get("query", "No query available")
        
        logger.warning(f"Handling SQL error: {error_message}")
        
        # Create an error response that's helpful to the user
        error_response = f"""
I'm sorry, but I encountered an issue while retrieving the data you requested. 

The database query could not be completed successfully. This might be because:
1. The specific data you're looking for might not be available in our database
2. There might be a technical issue with accessing the database
3. The way the request was phrased might not match our database structure

Error details: {error_message}

Would you like to try rephrasing your request or ask for different information?
"""
        
        # Update state
        state["response"] = error_response
        state["intermediate_steps"] = intermediate_steps
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