"""
Minimal example of using smolagents
"""

from smolagents import CodeAgent, tool, OpenAIServerModel, AgentImage, ToolCallingAgent
import os
from PIL import Image
import base64
from io import BytesIO
from openai import OpenAI

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# @tool
# def get_image() -> AgentImage:
#     """
#     Returns an image from the current directory.
        
#     Returns:
#         An AgentImage containing img.png from the current directory
#     """
#     # Return img.png from current directory as AgentImage
#     return AgentImage("img.png")

@tool
def analyze_image_with_llm() -> str:
    """
    Analyzes image with OpenAI API and writes the analysis to a file.
    
    Returns:
        A string containing the analysis of the image.
    """
    image_path = "./img.png"
    
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found.")
        return "Error: Image not found"
    
    print(f"Sending {image_path} to OpenAI for analysis...")
    
    # Load and encode the image
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    prompt_text = "Analyze this image. Describe what you see, including any objects, people, or significant features."
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        }
                    ],
                }
            ]
        )
        analysis_result = response.choices[0].message.content
        print("\n--- OpenAI Analysis ---")
        print(analysis_result)
        print("-----------------------\n")
        
        # Write the analysis to a file
        with open("analysis.txt", "w", encoding="utf-8") as f:
            f.write(analysis_result)
        print("Analysis saved to analysis.txt")
        
        return analysis_result
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return f"Error: {str(e)}"

def main():
    if "OPENAI_API_KEY" in os.environ:
        model = OpenAIServerModel(
            model_id="gpt-4o",
            api_base="https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"]
        )
    
    # Create the agent with the tool and model object
    agent = ToolCallingAgent(
        tools=[analyze_image_with_llm],
        model=model,
    )
    # agent = CodeAgent(
    #     tools=[get_image],
    #     model=model,
    # )
    
    # Run the agent with a simple task
    result = agent.run("Analyze the image and describe what you see.")
    print(f"Agent result: {result}")


if __name__ == "__main__":
    main()
