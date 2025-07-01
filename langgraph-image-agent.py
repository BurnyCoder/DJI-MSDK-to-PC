from langchain_core.tools import tool
from PIL import ImageGrab
from langgraph.prebuilt import create_react_agent
import base64

@tool
def get_screenshot():
    """
    Get a screenshot of the current screen
    """
    print("[get_screenshot] Taking screenshot...")
    screenshot = ImageGrab.grab()
    screenshot.save("screenshot.png")
    with open("screenshot.png", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_data}"},
        }
    ]
    

agent = create_react_agent(
    model="openai:gpt-4o-mini",  
    tools=[get_screenshot],  
    prompt="You are a helpful assistant that can take a screenshot of the current screen and analyze the image."  
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "What is screenshoted in the image?"}]}
)