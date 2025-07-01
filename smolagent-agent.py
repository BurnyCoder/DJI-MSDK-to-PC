"""
Minimal example of using smolagents
"""

from smolagents import CodeAgent, tool, OpenAIServerModel
import os

@tool
def greet(name: str) -> str:
    """
    Greets a person by name.
    
    Args:
        name: The name of the person to greet
        
    Returns:
        A greeting message
    """
    return f"Hello, {name}! Welcome to smolagents."


def main():
    if "OPENAI_API_KEY" in os.environ:
        model = OpenAIServerModel(
            model_id="gpt-4o-mini",
            api_base="https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"]
        )
    
    # Create the agent with the tool and model object
    agent = CodeAgent(
        tools=[greet],
        model=model,
    )
    
    # Run the agent with a simple task
    result = agent.run("Please greet Alice")
    print(f"Agent result: {result}")


if __name__ == "__main__":
    main()
