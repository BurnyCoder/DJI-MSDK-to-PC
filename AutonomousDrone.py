from smolagents import ToolCallingAgent, LiteLLMModel, tool, AgentImage
from OpenDJI import OpenDJI  # Ensure OpenDJI is installed and accessible
import os
from dotenv import load_dotenv
import numpy as np # For handling frame data like shape
from PIL import Image # For converting numpy array to PIL Image
import time # Added for delays in tracking loop
import base64 # Added for image encoding
from io import BytesIO # Added for image encoding
from openai import OpenAI # Added for OpenAI API
import atexit # Added for graceful connection closing

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Drone IP Address: fetched from .env or defaults if not set.
# Ensure your drone is connected to this IP address.
DRONE_IP_ADDR = os.getenv("DRONE_IP_ADDR", "192.168.1.115")
# OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- Global Drone Connection ---
drone_connection = None

def initialize_drone_connection():
    """Initializes the global drone connection."""
    global drone_connection
    if drone_connection is None:
        try:
            print(f"Attempting to connect to drone at IP: {DRONE_IP_ADDR}...")
            drone_connection = OpenDJI(DRONE_IP_ADDR)
            print("Successfully connected to the drone.")
            # Optionally, enable control right after connection
            # result = drone_connection.enableControl(True)
            # print(f"Enable SDK control attempt post-connection. Drone response: {result}")
        except Exception as e:
            print(f"Failed to connect to the drone: {e}")
            drone_connection = None # Ensure it's None on failure
            # raise ConnectionError(f"Failed to initialize drone connection: {e}") # Or raise an error
    return drone_connection

def close_drone_connection():
    """Closes the global drone connection if it's open."""
    global drone_connection
    if drone_connection:
        try:
            print("Closing drone connection...")
            # result = drone_connection.disableControl(True) # Optionally disable control before closing
            # print(f"Disable SDK control attempt pre-close. Drone response: {result}")
            drone_connection.close()
            print("Drone connection closed.")
        except Exception as e:
            print(f"Error closing drone connection: {e}")
        finally:
            drone_connection = None

# Register the close function to be called on exit
atexit.register(close_drone_connection)

# --- Drone Control Tools ---
@tool
def drone_takeoff() -> str:
    """
    Commands the drone to take off.

    Returns:
        str: A message indicating the result of the takeoff command.
    """
    global drone_connection
    if drone_connection is None:
        initialize_drone_connection()
        if drone_connection is None:
            return "Error: Drone connection not established. Cannot take off."
    try:
        result = drone_connection.enableControl(True)
        print(f"Enable SDK control command sent. Drone response: {result}")
        result = drone_connection.takeoff(True)
        return f"Takeoff command sent. Drone response: {result}"
    except Exception as e:
        return f"Error during takeoff: {str(e)}"

@tool
def drone_land() -> str:
    """
    Commands the drone to land.

    Returns:
        str: A message indicating the result of the land command.
    """
    global drone_connection
    if drone_connection is None:
        initialize_drone_connection()
        if drone_connection is None:
            return "Error: Drone connection not established. Cannot land."
    try:
        result = drone_connection.enableControl(True)
        print(f"Enable SDK control command sent. Drone response: {result}")
        result = drone_connection.land(True)
        return f"Land command sent. Drone response: {result}"
    except Exception as e:
        return f"Error during land: {str(e)}"

@tool
def move_drone(rcw: float, du: float, lr: float, bf: float) -> str:
    """
    Moves the drone with specified control values.

    These values are typically small floats, e.g., between -0.5 and 0.5.
    rcw: Rotational movement (rotate clockwise/anti-clockwise). Negative for anti-clockwise, positive for clockwise.
    du: Vertical movement (down/up). Negative for down, positive for up.
    lr: Sideways movement (left/right). Negative for left, positive for right.
    bf: Forward/backward movement. Negative for backward, positive for forward.

    Args:
        rcw: The rotation control value.
        du: The up/down control value.
        lr: The left/right control value.
        bf: The forward/backward control value.

    Returns:
        str: A message indicating the result of the move command.
    """
    global drone_connection
    if drone_connection is None:
        initialize_drone_connection()
        if drone_connection is None:
            return "Error: Drone connection not established. Cannot move."
    try:
        result = drone_connection.enableControl(True)
        print(f"Enable SDK control command sent. Drone response: {result}")
        drone_connection.move(rcw, du, lr, bf)
        return f"Move command sent: rcw={rcw}, du={du}, lr={lr}, bf={bf}"
    except Exception as e:
        return f"Error moving drone: {str(e)}"

@tool
def get_drone_frame_info() -> AgentImage:
    """
    Retrieves the current video frame from the drone as an AgentImage.

    Returns:
        AgentImage: An AgentImage object containing the frame, or a string with an error message if unsuccessful.
    """
    global drone_connection
    if drone_connection is None:
        initialize_drone_connection()
        if drone_connection is None:
            # This function is typed to return AgentImage, but error cases return str.
            # This inconsistency was in the original code.
            # For now, returning a string error message to match existing pattern.
            return "Error: Drone connection not established. Cannot get frame."
    try:
        # Assuming enableControl is not strictly needed for just getting a frame,
        # or if it is, it should be managed at a higher level for continuous operations.
        # If enableControl is needed here, it should be added:
        # result = drone_connection.enableControl(True)
        # print(f"Enable SDK control command sent for get_frame. Drone response: {result}")
        
        frame_np = drone_connection.getFrame() # Assuming this returns a NumPy array
        if frame_np is None:
            print("No frame available from the drone.")
            # Maintaining the string return type for error, as per original function's behavior.
            raise ValueError("No frame available from the drone.") 

        pil_image = Image.fromarray(frame_np.astype(np.uint8)) # Ensure correct dtype for PIL
        return AgentImage(pil_image)
    except ValueError as ve: # Catch the specific error for no frame
        return f"Error getting drone frame: {str(ve)}" 
    except Exception as e:
        return f"Error getting drone frame info: {str(e)}"

@tool
def track_person_and_rotate(max_iterations: int = 30, yaw_strength: float = 0.2, scan_yaw_strength: float = 0.015, seconds_per_iteration: float = 2.5, rotation_pulse_duration: float = 1.25) -> str:
    """
    Commands the drone to take off, then continuously looks for a person in the drone's video feed and rotates the drone to keep the person centered.
    Does not move the drone forward/backward/sideways or up/down during tracking.
    Rotation is applied in pulses. Finally, commands the drone to land.

    Args:
        max_iterations: The maximum number of tracking attempts.
        yaw_strength: The magnitude of yaw rotation when a person is detected off-center.
        scan_yaw_strength: The magnitude of yaw rotation when scanning for a person if not found.
        seconds_per_iteration: The target total cycle time for each iteration (includes processing, potential rotation, and waiting).
        rotation_pulse_duration: The duration (in seconds) for which the drone actively rotates before stopping, when a yaw adjustment is made.
    
    Returns:
        str: A message indicating the result of the tracking sequence.
    """
    print("Initiating automated person tracking sequence...")
    global drone_connection
    
    if drone_connection is None:
        initialize_drone_connection()
        if drone_connection is None:
            return "Error: Drone connection not established. Cannot start tracking."

    try:
        # Use the global drone_connection
        drone = drone_connection 
        
        result = drone.enableControl(True)
        print(f"Enable SDK control command sent. Drone response: {result}")
        # --- Takeoff --- 
        print("Sending takeoff command...")
        takeoff_result = drone.takeoff(True)
        print(f"Takeoff result: {takeoff_result}")
        if "error" in str(takeoff_result).lower() or "failed" in str(takeoff_result).lower():
            return f"Takeoff failed, cannot start tracking: {takeoff_result}"
        
        # Give a brief moment for the drone to stabilize after takeoff
        print("Stabilizing after takeoff...")
        time.sleep(5)

        print(f"Starting person tracking for up to {max_iterations} iterations.")
        person_sighted_in_previous_iteration = False
        consecutive_no_person_scans = 0 # Tracks how many consecutive frames a person isn't seen after being seen

        for i in range(max_iterations):
            print(f"Tracking iteration {i+1}/{max_iterations}...")
            current_yaw = 0.0
            iteration_logic_start_time = time.time()

            try:
                # Get frame directly from the drone instance
                frame_np = drone.getFrame()
                
                if frame_np is None:
                    print("No frame available from the drone. Skipping this iteration.")
                    time.sleep(seconds_per_iteration)
                    continue

                # Convert NumPy array to PIL Image
                pil_image = Image.fromarray(frame_np.astype(np.uint8))
                
                # Create an AgentImage for consistency with previous code
                agent_image = AgentImage(pil_image)

                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                # Using OpenAI for image analysis instead of LiteLLM
                print("Sending frame to OpenAI for analysis...")
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Analyze this image from a drone's camera. Is a person clearly visible? If yes, in which horizontal third of the image are they primarily located: 'left', 'center', or 'right'? If no person is clearly visible, or if their location cannot be reliably determined, respond with only one word: 'left', 'center', 'right', or 'none'."},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                            ]
                        }
                    ],
                    max_tokens=300,
                )
                
                llm_output = response.choices[0].message.content.strip().lower()
                print(f"OpenAI analysis result: '{llm_output}'")

                if "left" in llm_output:
                    current_yaw = -yaw_strength
                    print(f"Person detected on the left. Yawing left (strength: {current_yaw}).")
                    person_sighted_in_previous_iteration = True
                    consecutive_no_person_scans = 0
                elif "right" in llm_output:
                    current_yaw = yaw_strength
                    print(f"Person detected on the right. Yawing right (strength: {current_yaw}).")
                    person_sighted_in_previous_iteration = True
                    consecutive_no_person_scans = 0
                elif "center" in llm_output:
                    # current_yaw remains 0.0
                    print("Person detected in the center. Holding position.")
                    person_sighted_in_previous_iteration = True
                    consecutive_no_person_scans = 0
                    # Move forward if person is in the center
                    print(f"Moving forward towards person (strength: 0.1) for {rotation_pulse_duration}s...")
                    drone.move(0, 0, 0, 0.1) # bf = 0.1 for forward
                    time.sleep(rotation_pulse_duration)
                    drone.move(0, 0, 0, 0) # Stop forward movement
                else: # "none" or unexpected LLM output
                    print("No person clearly detected by OpenAI.")
                    if person_sighted_in_previous_iteration:
                        consecutive_no_person_scans += 1
                        if consecutive_no_person_scans <= 2:
                            print(f"Person lost (iteration {consecutive_no_person_scans} of being lost). Holding position to re-evaluate.")
                            # current_yaw remains 0.0
                        else:
                            print(f"Person lost for >2 iterations. Initiating scan.")
                            current_yaw = scan_yaw_strength * (-1 if (consecutive_no_person_scans - 3) % 2 == 0 else 1)
                            print(f"Scanning for person. Yaw: {current_yaw}")
                    else:
                        current_yaw = scan_yaw_strength * (-1 if i % 4 < 2 else 1) # Broader scan pattern: L, L, R, R
                        print(f"No person sighted previously. Scanning. Yaw: {current_yaw}")
                    
                    if not ("left" in llm_output or "right" in llm_output or "center" in llm_output): # If truly "none"
                        person_sighted_in_previous_iteration = False

                if current_yaw != 0.0:
                    # Move drone directly using the drone instance
                    print(f"Executing pulsed rotation: rcw={current_yaw} for {rotation_pulse_duration}s...")
                    drone.move(current_yaw, 0, 0, 0) # Start rotation
                    time.sleep(rotation_pulse_duration) # Rotate for the specified pulse duration
                    print("Stopping rotation.")
                    drone.move(0, 0, 0, 0) # Stop rotation
                else:
                    print("No yaw adjustment needed for this iteration.")

            except Exception as e:
                print(f"Error in tracking iteration {i+1}: {str(e)}")
            
            iteration_logic_end_time = time.time()
            time_spent_in_iteration_logic = iteration_logic_end_time - iteration_logic_start_time
            
            remaining_wait_time = seconds_per_iteration - time_spent_in_iteration_logic
            
            if remaining_wait_time > 0:
                print(f"Waiting for {remaining_wait_time:.2f} seconds before next iteration's processing...")
                time.sleep(remaining_wait_time)
            else:
                print(f"Iteration logic (processing/movement) took {time_spent_in_iteration_logic:.2f}s. Proceeding to next iteration immediately.")

        # --- Land --- 
        print("Landing the drone...")
        land_result = drone.land(True)
        print(f"Landing result: {land_result}")
        
        return f"Person tracking completed after {max_iterations} iterations."
    except Exception as e:
        print(f"An overall error occurred during the track_person_and_rotate sequence: {e}")
        return f"Error during tracking sequence: {e}"

# @tool
# def enable_drone_sdk_control() -> str:
#     """
#     Enables SDK control over the drone, allowing programmatic commands.
#     This typically disables manual control from a physical remote controller.
#     """
#     try:
#         with OpenDJI(DRONE_IP_ADDR) as drone:
#             result = drone.enableControl(True)
#             return f"Enable SDK control command sent. Drone response: {result}"
#     except Exception as e:
#         return f"Error enabling SDK control: {str(e)}"

# @tool
# def disable_drone_sdk_control() -> str:
#     """
#     Disables SDK control, returning control to the physical remote controller if available.
#     """
#     try:
#         with OpenDJI(DRONE_IP_ADDR) as drone:
#             result = drone.disableControl(True)
#             return f"Disable SDK control command sent. Drone response: {result}"
#     except Exception as e:
#         return f"Error disabling SDK control: {str(e)}"

class AutonomousDroneAgent:
    def __init__(self):
        # --- Agent Initialization ---
        # Initialize the LiteLLMModel
        # Using OpenRouter with Google Gemini.
        self.model = LiteLLMModel(
            model_id="openrouter/google/gemini-2.5-pro-exp-03-25",
            temperature=0.5,
            max_tokens=50000
        )

        # Ensure drone connection is initialized when agent is created
        initialize_drone_connection() 
        if drone_connection is None:
            # This is a critical failure for the agent if it relies on the drone.
            # Consider how to handle this - maybe raise an exception or log a severe warning.
            print("CRITICAL: Drone connection failed to initialize for AutonomousDroneAgent.")

        # Create the ToolCallingAgent with the defined drone tools
        # The tools are now defined outside the class
        self.drone_agent = ToolCallingAgent(
            tools=[
                drone_takeoff,
                drone_land,
                move_drone,
                get_drone_frame_info,
                #track_person_and_rotate,
                # enable_drone_sdk_control,
                # disable_drone_sdk_control
            ],
            model=self.model
        )

    def run_query(self, query: str) -> str:
        """
        Runs a query using the initialized drone agent.
        """
        print(f"Sending query to agent: '{query}'")
        try:
            response = self.drone_agent.run(query)
            print(f"Agent response:\n{response}")
            return response
        except Exception as e:
            error_message = f"Error running agent query: {e}"
            print(error_message)
            print("Please ensure the drone is connected and OpenDJI is set up correctly.")
            print(f"Also, check your OPENROUTER_API_KEY (if required) and DRONE_IP_ADDR (current: {DRONE_IP_ADDR}) in the .env file.")
            return error_message

# --- Example Usage (Optional) ---
if __name__ == "__main__":
    print("Initializing Autonomous Drone Agent...")
    try:
        # Attempt to initialize connection first, as it's critical
        if initialize_drone_connection() is None:
            print("Failed to connect to the drone. Aborting example usage.")
            # Optionally, exit here if connection is mandatory for any further steps
            # exit(1) 
        else:
            print(f"Drone connection established/verified at IP: {DRONE_IP_ADDR}")
            print("Ensure your drone is powered on and connected to the network.")

            # Initialize agent after ensuring (or attempting) connection
            agent_instance = AutonomousDroneAgent()
            print("Autonomous Drone Agent Initialized.")
            
            # Option 1: Use the agent to run queries
            print("\nOption 1: You can run queries against the agent instance. For example:")
            print("  response = agent_instance.run_query('Take off the drone.')")
            print("  print(response)")
            
            # Option 2: Directly use the person tracking function
            print("\nOption 2: Or directly use the person tracking function:")
            print("  result = track_person_and_rotate()")
            print("  print(result)")
            
            # Uncomment one of these to test (requires drone connection)
            print("\n--- Running Example ---")

            # Option 1 example:
            # query = "Get drone frame info."
            # agent_instance.run_query(query)
            
            # Option 2 example:
            result = track_person_and_rotate(max_iterations=100000)
            # print(result)

    except ValueError as ve:
        print(f"Initialization Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during initialization or example usage: {e}")
    finally:
        # Explicitly close connection here if not relying solely on atexit,
        # or if specific cleanup order is needed before other atexit handlers.
        # However, atexit should handle it.
        # close_drone_connection() # This might be redundant due to atexit
        print("Application finished.")
