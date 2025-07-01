"""
This application flies a DJI drone forward for 5 seconds, capturing one frame per second.
It then sends these frames to OpenAI for analysis.
"""

from OpenDJI import OpenDJI
import time
import os
import base64
from io import BytesIO
from openai import OpenAI
from PIL import Image
import numpy as np
from smolagents import CodeAgent, tool, OpenAIServerModel
from typing import Union

# IP address of the connected android device
# IP_ADDR = os.environ.get("IP_ADDR", "100.93.47.145")
IP_ADDR = os.environ.get("IP_ADDR", "100.85.47.22")

# OpenAI API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize OpenAI client
if not OPENAI_API_KEY:
    print("OPENAI_API_KEY environment variable not found. OpenAI functionality will be disabled.")
    openai_client = None
else:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized successfully.")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        openai_client = None

# Movement factor
MOVE_VALUE = float(os.environ.get("MOVE_VALUE", "0.1"))


def _call_openai_and_save_analysis(prompt_messages, openai_client):
    """
    Helper function to call OpenAI API and save analysis to file.
    
    Args:
        prompt_messages: The prompt messages to send to OpenAI
        openai_client: The OpenAI client instance
    
    Returns:
        The analysis result string, or None if there was an error
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt_messages,
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
        return None


def _image_to_base64(pil_image):
    """
    Converts a PIL Image to base64 string.
    
    Args:
        pil_image: PIL Image object
    
    Returns:
        Base64 encoded string of the image
    """
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def capture_and_save_frame(drone, frame_num, timestamp, frames_dir="frames"):
    """
    Captures a frame from the drone and saves it immediately.
    
    Args:
        drone: The drone object
        frame_num: Frame number for naming
        timestamp: Timestamp for unique filename
        frames_dir: Directory to save frames
    
    Returns:
        A tuple (frame, file_path) if successful, (None, None) otherwise
    """
    # Create directory if it doesn't exist
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    frame = drone.getFrame()
    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
        print(f"Frame {frame_num} captured.")
        
        # Save the frame immediately
        try:
            pil_image = Image.fromarray(frame.astype(np.uint8))
            file_path = os.path.join(frames_dir, f"frame_{timestamp}_{frame_num}.png")
            pil_image.save(file_path)
            print(f"Frame {frame_num} saved to {file_path}")
            return frame, file_path
        except Exception as e:
            print(f"Error saving frame {frame_num}: {e}")
            return frame, None
    else:
        print(f"Could not capture frame {frame_num}.")
        return None, None


def analyze_frames_with_llm(captured_frames, openai_client):
    """
    Analyzes a list of frames with OpenAI API and writes the analysis to a file.
    """
    if not captured_frames:
        print("No frames to analyze.")
        return

    print(f"Sending {len(captured_frames)} frames to OpenAI for analysis...")
    
    base64_frames = []
    for frame in captured_frames:
        pil_image = Image.fromarray(frame.astype(np.uint8))
        img_base64 = _image_to_base64(pil_image)
        base64_frames.append(img_base64)

    prompt_text = f"Analyze these frames from a drone's flight. Describe what you see, including any objects, people, or significant features in the environment. Note any changes or movement across the frames."
    
    prompt_messages = [
        {
            "type": "text",
            "text": prompt_text,
        }
    ]
    for b64_frame in base64_frames:
        prompt_messages.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_frame}"}
            }
        )

    _call_openai_and_save_analysis(prompt_messages, openai_client)

def analyze_frame_with_llm(image_name: str, prompt: str) -> str:
    """
    Analyzes image with OpenAI API and writes the analysis to a file.
    
    Returns:
        A string containing the analysis of the image.
    """
    
    if not os.path.exists(image_name):
        print(f"Image {image_name} not found.")
        return "Error: Image not found"
    
    print(f"Sending {image_name} to OpenAI for analysis...")
    
    # Load and encode the image
    with Image.open(image_name) as img:
        img_base64 = _image_to_base64(img)
    
    # Use the provided prompt if given, otherwise use default
    prompt_text = prompt if prompt else "Analyze this image. Describe what you see, including any objects, people, or significant features."
    
    prompt_messages = [
        {
            "type": "text",
            "text": prompt_text,
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
        }
    ]

    analysis_result = _call_openai_and_save_analysis(prompt_messages, openai_client)
    
    if analysis_result is None:
        return "Error: Failed to analyze image"
    return analysis_result

class DroneAgentController:
    def __init__(self, drone, openai_client):
        self.drone = drone
        self.openai_client = openai_client
        self.captured_frames = []
        
        result = self.drone.enableControl(True)
        print(f"Enable SDK control command sent. Drone response: {result}")

    def takeoff(self):
        """Takes off the drone and hovers for 10 seconds."""
        print("Taking off...")
        self.drone.takeoff(True)
        time.sleep(10)
        print("Takeoff complete.")
        return "Takeoff complete."

    def land(self):
        """Lands the drone."""
        print("Landing drone...")
        self.drone.land(True)
        print("Drone landed.")
        return "Drone landed."

    def fly_forward(self):
        """
        Flies the drone forward for 10 seconds and captures one frame per second.
        The captured frames are stored internally for later analysis.
        Returns the path to the last captured frame.
        """
        flight_duration = 7
        print(f"Starting {flight_duration}-second forward flight and capturing frames.")
        captured_frames = []
        last_image_path = None
        
        # Generate timestamp for unique filenames
        timestamp = int(time.time())

        # Capture frame at the start of flying
        time.sleep(1)  # Wait before capturing
        start_frame, start_path = capture_and_save_frame(self.drone, len(captured_frames) + 1, timestamp)
        if start_frame is not None:
            captured_frames.append(start_frame)
            if start_path:
                last_image_path = start_path
            print("Captured frame at start of flight.")
        time.sleep(1)  # Wait after capturing

        # Loop for flight_duration seconds, sending command every 20ms
        command_interval = 0.02  # seconds
        num_iterations = int(flight_duration / command_interval)

        for i in range(num_iterations):
            self.drone.move(pitch=MOVE_VALUE, yaw=0, roll=0, ascent=0)
            time.sleep(command_interval)

        # Stop the drone
        self.drone.move(0, 0, 0, 0)
        
        # Capture frame at the end of flying
        time.sleep(1)  # Wait before capturing
        end_frame, end_path = capture_and_save_frame(self.drone, len(captured_frames) + 1, timestamp)
        if end_frame is not None:
            captured_frames.append(end_frame)
            if end_path:
                last_image_path = end_path
            print("Captured frame at end of flight.")
        time.sleep(1)  # Wait after capturing
        
        print(f"Forward flight complete. Captured {len(captured_frames)} frames.")
        self.captured_frames.extend(captured_frames)
        
        # Return the last captured image path
        if last_image_path:
            return last_image_path
        else:
            return "Forward flight complete but no frames were saved."

    def rotate_90_degrees(self):
        """Rotates the drone 90 degrees clockwise and captures frames at start and end.
        Returns the path to the last captured frame."""
        print("Rotating 90 degrees...")
        rotation_duration = 1.4  # seconds, adjusted from 2.0 to fix 120->90 degree issue
        command_interval = 0.02  # seconds
        num_iterations = int(rotation_duration / command_interval)
        
        # Generate timestamp for unique filenames
        timestamp = int(time.time())
        rotation_frames = []
        last_image_path = None
        
        # Capture frame at the start of rotation
        time.sleep(1)  # Wait before capturing
        start_frame, start_path = capture_and_save_frame(self.drone, len(self.captured_frames) + 1, timestamp)
        if start_frame is not None:
            rotation_frames.append(start_frame)
            if start_path:
                last_image_path = start_path
            print("Captured frame at start of rotation.")
        time.sleep(1)  # Wait after capturing

        for i in range(num_iterations):
            self.drone.move(pitch=0, yaw=1.0, roll=0, ascent=0)
            time.sleep(command_interval)
        
        # Stop rotation
        self.drone.move(0, 0, 0, 0)
        
        # Capture frame at the end of rotation
        time.sleep(1)  # Wait before capturing
        end_frame, end_path = capture_and_save_frame(self.drone, len(self.captured_frames) + len(rotation_frames) + 1, timestamp)
        if end_frame is not None:
            rotation_frames.append(end_frame)
            if end_path:
                last_image_path = end_path
            print("Captured frame at end of rotation.")
        time.sleep(1)  # Wait after capturing
        
        print(f"Rotation complete. Captured {len(rotation_frames)} frames.")
        
        # Add rotation frames to the main captured_frames list
        self.captured_frames.extend(rotation_frames)
        
        # Return the last captured image path
        if last_image_path:
            return last_image_path
        else:
            return "Rotation complete but no frames were saved."

    def analyze_captured_frames(self):
        """
        Analyzes the frames captured during the flight using OpenAI.
        This tool must be used after flying and capturing frames.
        """
        if not self.captured_frames:
            return "No frames have been captured yet. Use fly_forward_and_capture_frames first."
        
        analyze_frames_with_llm(
            self.captured_frames, 
            self.openai_client, 
        )
        return "Analysis is complete. The result has been saved to analysis.txt."


def run_test_flight_pattern(controller: DroneAgentController):
    """Runs a pre-defined flight pattern for testing."""
    print("Running test flight pattern...")
    print("1. Flying forward.")
    controller.fly_forward()
    print("2. Flying forward again.")
    controller.fly_forward()
    print("3. Rotating 90 degrees.")
    controller.rotate_90_degrees()
    print("4. Flying forward.")
    controller.fly_forward()
    print("5. Rotating 90 degrees.")
    controller.rotate_90_degrees()
    print("Test flight pattern complete.")


# Connect to the drone
with OpenDJI(IP_ADDR) as drone:
    if openai_client is None:
        print("OpenAI client not initialized. Exiting application.")
    else:
        controller = DroneAgentController(drone, openai_client)

        # Define tools after controller is created
        @tool
        def fly_forward() -> str:
            """
            Flies the drone forward for 10 seconds and captures frames at start and end.
            
            Returns:
                The file path of the last captured frame (e.g., 'frames/frame_1234567890_2.png').
                This path can be directly used with the analyze_frame tool.
            """
            result = controller.fly_forward()
            print(f"fly_forward returned: {result}")  # Add logging for clarity
            return result

        @tool
        def rotate_90_degrees() -> str:
            """
            Rotates the drone 90 degrees clockwise and captures frames at start and end.
            
            Returns:
                The file path of the last captured frame (e.g., 'frames/frame_1234567890_2.png').
                This path can be directly used with the analyze_frame tool.
            """
            result = controller.rotate_90_degrees()
            print(f"rotate_90_degrees returned: {result}")  # Add logging for clarity
            return result

        @tool
        def analyze_captured_frames() -> str:
            """
            Analyzes the frames captured during the flight using LLM. 
            """
            return controller.analyze_captured_frames()
        
        @tool 
        def analyze_frame(image_name: str, prompt: str) -> str:
            """
            Analyzes a single frame with LLM.
            
            Args:
                image_name: The file path to the image (e.g., 'frames/frame_1234567890_2.png')
                prompt: The analysis prompt to use
                
            Returns:
                The analysis result as a string
            """
            return analyze_frame_with_llm(image_name, prompt)

        tools = [
            fly_forward,
            rotate_90_degrees,
            # analyze_captured_frames,
            analyze_frame,
        ]

        # Create the model object
        model = OpenAIServerModel(
            model_id="gpt-4o",
            api_base="https://api.openai.com/v1",
            api_key=OPENAI_API_KEY
        )

        agent = CodeAgent(
            tools=tools,
            model=model,
        )

        # Takeoff before executing agent tasks
        controller.takeoff()

        # run_test_flight_pattern(controller)

        #prompt = "You are a drone operations agent. Your goal is to safely fly a drone to collect visual data and then analyze it and give a report to the user. The drone is already flying. Your task is to fly forward, rotate, fly forward, rotate fly forward, and then give report back to the user about what you've seen. Execute each task one by one and analyze each image and tell the user whats in the images."
#         prompt = """You are a drone operations agent. Your goal is to safely fly a drone to collect visual data and analyze it and give a report to the user. The drone is already flying. 

# Your task is to:
# 1. Fly forward (this will return an image path) and analyze that image
# 2. Rotate 90 degrees (this will return an image path) and analyze that image  
# 3. Fly forward (this will return an image path) and analyze that image
# 4. Rotate 90 degrees (this will return an image path) and analyze that image
# 5. Fly forward (this will return an image path) and analyze that image
# 6. Give a comprehensive report back to the user about what you've seen

# Important: The fly_forward and rotate_90_degrees functions return the file path of the captured image. You must use this returned path with the analyze_frame function."""
        
        prompt = """You are a drone operations agent. Your goal is to safely fly a drone to collect visual data and analyze it and give a report to the user. The drone is already flying. Your task is explore the area with calling the fly_forward and rotate_90_degrees functions as you see fit, and then give report back to the user about what you've seen. To analyze the images, you can use the analyze_frame function, where you can specify the image path and a prompt, so make sure to ask what details you want in the prompt, for example suggestions where to look or fly. Maximal amount of function calls is 10, minimum is 5. """
        
        # prompt = """You are a drone operations agent. Your goal is to safely fly a drone to collect visual data and analyze it. The drone is already flying. Your task is explore the area with calling the fly_forward and rotate_90_degrees functions as you see fit, and find a person and give a description of that person. To analyze the images, you can use the analyze_frame function, where you can specify the image path and a prompt, so make sure to ask what details you want in the prompt, for example suggestions where to look or fly. Maximal amount of function calls is 10, minimum is 5. """
        
        print(f"Agent executing prompt: '{prompt}'")
        agent.run(prompt)
        
        # Land after executing agent tasks
        controller.land()


