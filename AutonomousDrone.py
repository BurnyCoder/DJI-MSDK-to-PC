from smolagents import ToolCallingAgent, LiteLLMModel, tool
from OpenDJI import OpenDJI  # Ensure OpenDJI is installed and accessible
import os
from dotenv import load_dotenv
import numpy as np # For handling frame data like shape

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Drone IP Address: fetched from .env or defaults if not set.
# Ensure your drone is connected to this IP address.
DRONE_IP_ADDR = os.getenv("DRONE_IP_ADDR", "192.168.1.115")

# Anthropic API Key for the LLM model
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please set it in your .env file.")

# --- Drone Control Tools ---

@tool
def drone_takeoff() -> str:
    """Commands the drone to take off."""
    try:
        with OpenDJI(DRONE_IP_ADDR) as drone:
            result = drone.takeoff(True)
            return f"Takeoff command sent. Drone response: {result}"
    except Exception as e:
        return f"Error during takeoff: {str(e)}"

@tool
def drone_land() -> str:
    """Commands the drone to land."""
    try:
        with OpenDJI(DRONE_IP_ADDR) as drone:
            result = drone.land(True)
            return f"Land command sent. Drone response: {result}"
    except Exception as e:
        return f"Error during land: {str(e)}"

@tool
def move_drone(yaw: float, ascent: float, roll: float, pitch: float) -> str:
    """
    Moves the drone with specified control values.
    These values are typically small floats, e.g., between -0.5 and 0.5.
    Yaw: Rotational movement (left/right). Negative for left, positive for right.
    Ascent: Vertical movement (up/down). Negative for down, positive for up.
    Roll: Sideways movement (left/right). Negative for left, positive for right.
    Pitch: Forward/backward movement. Negative for backward, positive for forward.
    Args:
        yaw: The yaw control value.
        ascent: The ascent control value.
        roll: The roll control value.
        pitch: The pitch control value.
    """
    try:
        with OpenDJI(DRONE_IP_ADDR) as drone:
            # Ensure drone control is enabled for movement commands
            # drone.enableControl(True) # Optional: ensure control is enabled before move
            drone.move(yaw, ascent, roll, pitch)
            return f"Move command sent: yaw={yaw}, ascent={ascent}, roll={roll}, pitch={pitch}"
    except Exception as e:
        return f"Error moving drone: {str(e)}"

@tool
def get_drone_frame_info() -> str:
    """
    Retrieves information about the current video frame from the drone,
    such as its dimensions and data type. Does not return the image data itself.
    """
    try:
        with OpenDJI(DRONE_IP_ADDR) as drone:
            frame = drone.getFrame()
            if frame is None:
                return "No frame available from the drone."
            return f"Frame received: shape={frame.shape}, dtype={frame.dtype}"
    except Exception as e:
        return f"Error getting drone frame info: {str(e)}"

@tool
def enable_drone_sdk_control() -> str:
    """
    Enables SDK control over the drone, allowing programmatic commands.
    This typically disables manual control from a physical remote controller.
    """
    try:
        with OpenDJI(DRONE_IP_ADDR) as drone:
            result = drone.enableControl(True)
            return f"Enable SDK control command sent. Drone response: {result}"
    except Exception as e:
        return f"Error enabling SDK control: {str(e)}"

@tool
def disable_drone_sdk_control() -> str:
    """
    Disables SDK control, returning control to the physical remote controller if available.
    """
    try:
        with OpenDJI(DRONE_IP_ADDR) as drone:
            result = drone.disableControl(True)
            return f"Disable SDK control command sent. Drone response: {result}"
    except Exception as e:
        return f"Error disabling SDK control: {str(e)}"

# --- Agent Initialization ---

# Initialize the LiteLLMModel
# Using Claude 3.5 Sonnet as per the user's example
model = LiteLLMModel(
    model_id="claude-3-5-sonnet-20240620",
    api_key=ANTHROPIC_API_KEY
)

# Create the ToolCallingAgent with the defined drone tools
drone_agent = ToolCallingAgent(
    tools=[
        drone_takeoff,
        drone_land,
        move_drone,
        get_drone_frame_info,
        enable_drone_sdk_control,
        disable_drone_sdk_control
    ],
    model=model
)

# --- Example Usage (Optional) ---
if __name__ == "__main__":
    print("Autonomous Drone Agent Initialized.")
    print(f"Attempting to connect to drone at IP: {DRONE_IP_ADDR}")
    print("Ensure your drone is powered on and connected to the network.")
    print("You can now run queries against the 'drone_agent'. For example:")
    print("  response = drone_agent.run('Take off the drone.')")
    print("  print(response)")
    print("\nOr a more complex command:")
    print("  response = drone_agent.run('Enable SDK control, then take off, move forward a bit, then land, and finally disable SDK control.')")
    print("  print(response)")

    # Example of running a query (uncomment to test, requires drone connection)
    # print("\n--- Example Agent Query ---")
    # try:
    #     # Ensure drone is ready and it's safe to take off before uncommenting.
    #     # query = "Take off the drone, then get frame info, then land the drone."
    #     query = "Get drone frame info." # A safer first command
    #     print(f"Sending query to agent: '{query}'")
    #     response = drone_agent.run(query)
    #     print(f"Agent response:\n{response}")
    # except Exception as e:
    #     print(f"Error running agent query: {e}")
    #     print("Please ensure the drone is connected and OpenDJI is set up correctly.")
    #     print("Also, check your ANTHROPIC_API_KEY and DRONE_IP_ADDR in the .env file.")

    pass
