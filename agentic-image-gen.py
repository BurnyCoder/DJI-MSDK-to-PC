import openai
import requests
from datetime import datetime
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Set your OpenAI API key
# openai.api_key = "YOUR_OPENAI_API_KEY"

@tool
def generate_image(prompt: str) -> str:
    """Generates an image from a text prompt using DALL-E and returns the image URL."""
    try:
        response = openai.images.generate(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url
        
        # Download and save the image
        img_response = requests.get(image_url)
        if img_response.status_code == 200:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_image_{timestamp}.png"
            
            # Save the image
            with open(filename, 'wb') as f:
                f.write(img_response.content)
            
            return f"Image saved as '{filename}'. URL: {image_url}"
        else:
            return f"Image generated but could not be saved. URL: {image_url}"
            
    except Exception as e:
        return f"An error occurred during image generation: {e}"

# Initialize the language model
llm = ChatOpenAI(temperature=0)

# Define the agent's prompt
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can generate images."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
tools = [generate_image]
agent = create_openai_tools_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
response = agent_executor.invoke({
    "input": "Generate an image of a futuristic city with flying cars."
})

print(response["output"])