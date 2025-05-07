from smolagents import ToolCallingAgent, LiteLLMModel, tool, AgentImage
from OpenDJI import OpenDJI  # Ensure OpenDJI is installed and accessible
import os
from dotenv import load_dotenv
import numpy as np # For handling frame data like shape
from PIL import Image # For converting numpy array to PIL Image
import time # Added for delays in tracking loop
import litellm # Added for direct multimodal LLM calls
import base64 # Added for image encoding
from io import BytesIO # Added for image encoding

# Load environment variables from .env file
load_dotenv()

class AutonomousDroneAgent:
    def __init__(self):
        # --- Configuration ---
        # Drone IP Address: fetched from .env or defaults if not set.
        # Ensure your drone is connected to this IP address.
        self.DRONE_IP_ADDR = os.getenv("DRONE_IP_ADDR", "192.168.1.115")

        # --- Agent Initialization ---
        # Initialize the LiteLLMModel
        # Using OpenRouter with Google Gemini.
        self.model = LiteLLMModel(
            model_id="openrouter/google/gemini-2.5-pro-exp-03-25",
            temperature=0.5,
            max_tokens=50000
        )

        # Create the ToolCallingAgent with the defined drone tools
        # The tools will be methods of this class, decorated with @tool
        self.drone_agent = ToolCallingAgent(
            tools=[
                self.drone_takeoff,
                self.drone_land,
                self.move_drone,
                self.get_drone_frame_info,
                #self.track_person_and_rotate,
                # self.enable_drone_sdk_control,
                # self.disable_drone_sdk_control
            ],
            model=self.model
        )

    # --- Drone Control Tools ---
    @tool
    def drone_takeoff(self) -> str:
        """Commands the drone to take off."""
        try:
            with OpenDJI(self.DRONE_IP_ADDR) as drone:
                result = drone.takeoff(True)
                return f"Takeoff command sent. Drone response: {result}"
        except Exception as e:
            return f"Error during takeoff: {str(e)}"

    @tool
    def drone_land(self) -> str:
        """Commands the drone to land."""
        try:
            with OpenDJI(self.DRONE_IP_ADDR) as drone:
                result = drone.land(True)
                return f"Land command sent. Drone response: {result}"
        except Exception as e:
            return f"Error during land: {str(e)}"

    @tool
    def move_drone(self, yaw: float, ascent: float, roll: float, pitch: float) -> str:
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
            with OpenDJI(self.DRONE_IP_ADDR) as drone:
                # Ensure drone control is enabled for movement commands
                # drone.enableControl(True) # Optional: ensure control is enabled before move
                drone.move(yaw, ascent, roll, pitch)
                return f"Move command sent: yaw={yaw}, ascent={ascent}, roll={roll}, pitch={pitch}"
        except Exception as e:
            return f"Error moving drone: {str(e)}"

    @tool
    def get_drone_frame_info(self) -> AgentImage:
        """
        Retrieves the current video frame from the drone as an AgentImage.
        """
        try:
            with OpenDJI(self.DRONE_IP_ADDR) as drone:
                frame_np = drone.getFrame() # Assuming this returns a NumPy array
                if frame_np is None:
                    print("No frame available from the drone.")
                    raise "No frame available from the drone."

                # Convert NumPy array to PIL Image
                # Assuming frame_np is in a format that PIL can understand (e.g., RGB, Grayscale)
                # If the drone returns BGR, it might need conversion: pil_image = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
                # For now, assuming direct conversion works.
                pil_image = Image.fromarray(frame_np.astype(np.uint8)) # Ensure correct dtype for PIL
                return AgentImage(pil_image)
        except ValueError as ve: # Catch the specific error for no frame
            return f"Error getting drone frame: {str(ve)}" # Return string on error
        except Exception as e:
            # This also returns a string, which is inconsistent with AgentImage return type.
            # This function needs a consistent error handling strategy for AgentImage.
            # For now, following the existing pattern of returning error strings.
            return f"Error getting drone frame info: {str(e)}"

    @tool
    def track_person_and_rotate(self, max_iterations: int = 30, yaw_strength: float = 0.2, scan_yaw_strength: float = 0.15, seconds_per_iteration: float = 2.5) -> str:
        """
        Continuously looks for a person in the drone's video feed and rotates the drone to keep the person centered.
        Does not move the drone forward/backward/sideways or up/down.
        Args:
            max_iterations: The maximum number of tracking attempts.
            yaw_strength: The magnitude of yaw rotation when a person is detected off-center.
            scan_yaw_strength: The magnitude of yaw rotation when scanning for a person if not found.
            seconds_per_iteration: Time to wait between iterations, allowing for drone movement, new frame capture, and LLM analysis.
        """
        print(f"Starting person tracking for up to {max_iterations} iterations.")
        person_sighted_in_previous_iteration = False
        consecutive_no_person_scans = 0 # Tracks how many consecutive frames a person isn't seen after being seen

        for i in range(max_iterations):
            print(f"Tracking iteration {i+1}/{max_iterations}...")
            current_yaw = 0.0
            try:
                frame_result = self.get_drone_frame_info()

                if isinstance(frame_result, str):
                    print(f"Could not get drone frame: {frame_result}. Skipping this iteration.")
                    time.sleep(seconds_per_iteration)
                    continue

                agent_image = frame_result
                pil_image = agent_image.pil_image

                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this image from a drone's camera. Is a person clearly visible? If yes, in which horizontal third of the image are they primarily located: 'left', 'center', or 'right'? If no person is clearly visible, or if their location cannot be reliably determined, respond with only one word: 'left', 'center', 'right', or 'none'."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                        ]
                    }
                ]
                
                print("Sending frame to LLM for analysis...")
                response = litellm.completion(
                    model=self.model.model_id,
                    messages=messages,
                    temperature=0.2, 
                    max_tokens=50 
                )
                
                llm_output = response.choices[0].message.content.strip().lower()
                print(f"LLM analysis result: '{llm_output}'")

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
                else: # "none" or unexpected LLM output
                    print("No person clearly detected by LLM.")
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
                    move_result = self.move_drone(yaw=current_yaw, ascent=0, roll=0, pitch=0)
                    print(f"Move command result: {move_result}")
                else:
                    print("No yaw adjustment needed for this iteration.")

            except litellm.exceptions.APIConnectionError as e:
                print(f"LLM API Connection Error: {str(e)}. Check API key, network, and model availability. Skipping iteration.")
            except litellm.exceptions.RateLimitError as e:
                print(f"LLM Rate Limit Error: {str(e)}. Waiting before retrying or skipping. Skipping iteration.")
            except Exception as e:
                print(f"Error in tracking iteration {i+1}: {str(e)}")
            
            print(f"Waiting for {seconds_per_iteration} seconds before next iteration...")
            time.sleep(seconds_per_iteration)

        return f"Person tracking completed after {max_iterations} iterations."

    # @tool
    # def enable_drone_sdk_control(self) -> str:
    #     """
    #     Enables SDK control over the drone, allowing programmatic commands.
    #     This typically disables manual control from a physical remote controller.
    #     """
    #     try:
    #         with OpenDJI(self.DRONE_IP_ADDR) as drone:
    #             result = drone.enableControl(True)
    #             return f"Enable SDK control command sent. Drone response: {result}"
    #     except Exception as e:
    #         return f"Error enabling SDK control: {str(e)}"

    # @tool
    # def disable_drone_sdk_control(self) -> str:
    #     """
    #     Disables SDK control, returning control to the physical remote controller if available.
    #     """
    #     try:
    #         with OpenDJI(self.DRONE_IP_ADDR) as drone:
    #             result = drone.disableControl(True)
    #             return f"Disable SDK control command sent. Drone response: {result}"
    #     except Exception as e:
    #         return f"Error disabling SDK control: {str(e)}"

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
            print(f"Also, check your OPENROUTER_API_KEY (if required) and DRONE_IP_ADDR (current: {self.DRONE_IP_ADDR}) in the .env file.")
            return error_message

# --- Example Usage (Optional) ---
if __name__ == "__main__":
    print("Initializing Autonomous Drone Agent...")
    try:
        agent_instance = AutonomousDroneAgent()
        print("Autonomous Drone Agent Initialized.")
        print(f"Attempting to connect to drone at IP: {agent_instance.DRONE_IP_ADDR}")
        print("Ensure your drone is powered on and connected to the network.")
        
        # Option 1: Use the agent to run queries
        print("\nOption 1: You can run queries against the agent instance. For example:")
        print("  response = agent_instance.run_query('Take off the drone.')")
        print("  print(response)")
        
        # Option 2: Directly use the person tracking function
        print("\nOption 2: Or directly use the person tracking function:")
        print("  result = agent_instance.track_person_and_rotate()")
        print("  print(result)")
        
        # Uncomment one of these to test (requires drone connection)
        print("\n--- Running Example ---")

        # Option 1 example:
        # query = "Get drone frame info."
        # agent_instance.run_query(query)
        
        # Option 2 example:
        result = agent_instance.track_person_and_rotate(max_iterations=100000)
        # print(result)

    except ValueError as ve:
        print(f"Initialization Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during initialization or example usage: {e}")
