"""
Minimal example of using smolagents
"""

from smolagents import CodeAgent, tool, OpenAIServerModel, AgentImage
import os

@tool
def get_image(name: str) -> AgentImage:
    """
    Returns an image from the current directory.
    
    Args:
        name: The name parameter (not used in this example)
        
    Returns:
        An AgentImage containing img.png from the current directory
    """
    # Return img.png from current directory as AgentImage
    return AgentImage("img.png")


def main():
    if "OPENAI_API_KEY" in os.environ:
        model = OpenAIServerModel(
            model_id="gpt-4o",
            api_base="https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"]
        )
    
    # Create the agent with the tool and model object
    agent = CodeAgent(
        tools=[get_image],
        model=model,
    )
    
    # Run the agent with a simple task
    result = agent.run("Analyze the image and describe what you see.")
    print(f"Agent result: {result}")


if __name__ == "__main__":
    main()
