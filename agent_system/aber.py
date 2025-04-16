import os
import logging
from typing import Dict, Any, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ChatMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI  # Import the actual Gemini class

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class DirectorAgent:
    """Simplified DirectorAgent for testing Gemini initialization."""
    def __init__(self, llm):
        self.llm = llm
        self.intent_prompt = """
You are a Director Agent. Categorize the user request: {user_input}
"""
        self.synthesis_prompt = "..."  # Not needed for this test

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        intermediate_steps = state.get("intermediate_steps", [])

        logger.info("===== PROCESSING INITIAL REQUEST =====")
        formatted_prompt = self.intent_prompt.format(user_input=user_input)
        logger.info(f"Intent Prompt: {formatted_prompt}")

        try:
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages).content # Here is the change
            logger.info(f"Intent Response: {response}")

            state["response"] = response
            state["current_agent"] = "director"
            intermediate_steps.append({
                "agent": "director",
                "action": "analyze_intent",
                "input": user_input,
                "output": response,
                "timestamp": "mock_timestamp"
            })
            state["intermediate_steps"] = intermediate_steps
            return state

        except Exception as e:
            logger.error(f"Error in Director Agent with Gemini: {e}")
            state["response"] = f"Error with Gemini: {e}"
            return state


if __name__ == "__main__":
    # Use hardcoded values for testing
    api_key = os.environ.get("GOOGLE_API_KEY")  # IMPORTANT: Use the API key from the environment
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in the environment.")
    model = "gemini-2.0-flash"  # Use "gemini-2.0-flash" here
    temperature = 0.1

    gemini_llm = ChatGoogleGenerativeAI(  # Initialize the REAL Gemini LLM
        api_key=api_key,
        model=model,
        temperature=temperature,
        convert_system_message_to_human=True
    )
    logger.info("Gemini LLM Initialized")

    director_agent = DirectorAgent(gemini_llm)

    # Simulate an incoming user request
    test_state = {"user_input": "Test Gemini initialization."}
    updated_state = director_agent(test_state)
    print("\n===== Updated State =====")
    print(updated_state)
