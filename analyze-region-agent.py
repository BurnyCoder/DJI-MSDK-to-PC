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
from smolagents import CodeAgent, tool

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


def fly_forward_and_capture(drone, flight_duration=10):
    """
    Flies the drone forward for a specified duration and captures frames every second.
    """
    print(f"Starting {flight_duration}-second forward flight and capturing frames.")
    captured_frames = []

    # Loop for flight_duration seconds, sending command every 20ms
    command_interval = 0.02  # seconds
    num_iterations = int(flight_duration / command_interval)
    capture_interval = int(1 / command_interval)

    for i in range(num_iterations):
        drone.move(pitch=MOVE_VALUE, yaw=0, roll=0, ascent=0)

        # Capture a frame every second
        if (i + 1) % capture_interval == 0:
            frame = drone.getFrame()
            if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                captured_frames.append(frame)
                print(f"Frame {len(captured_frames)} captured.")
            else:
                print(f"Could not capture frame {len(captured_frames) + 1}.")
        
        time.sleep(command_interval)

    # Stop the drone
    drone.move(0, 0, 0, 0)
    print(f"Forward flight complete. Captured {len(captured_frames)} frames.")
    return captured_frames


def analyze_frames_with_openai(captured_frames, openai_client, flight_duration):
    """
    Analyzes a list of frames with OpenAI API, saves them, and writes the analysis to a file.
    """
    if not captured_frames:
        print("No frames to analyze.")
        return

    # Save frames to files with timestamp to avoid overriding
    frames_dir = "forward_analysis_frames"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    # Generate timestamp for unique filenames
    timestamp = int(time.time())
    
    print(f"Saving {len(captured_frames)} frames to '{frames_dir}' directory...")
    for i, frame in enumerate(captured_frames):
        try:
            pil_image = Image.fromarray(frame.astype(np.uint8))
            file_path = os.path.join(frames_dir, f"frame_{timestamp}_{i+1}.png")
            pil_image.save(file_path)
        except Exception as e:
            print(f"Error saving frame {i+1}: {e}")
    print("All frames saved.")

    print("Sending frames to OpenAI for analysis...")
    
    base64_frames = []
    for frame in captured_frames:
        pil_image = Image.fromarray(frame.astype(np.uint8))
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        base64_frames.append(img_base64)

    prompt_text = f"Analyze these frames from a drone's {flight_duration}-second forward flight. Describe what you see, including any objects, people, or significant features in the environment. Note any changes or movement across the frames."
    
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

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt_messages,
                }
            ],
            max_tokens=500,
        )
        analysis_result = response.choices[0].message.content
        print("\n--- OpenAI Analysis ---")
        print(analysis_result)
        print("-----------------------\n")
        
        # Write the analysis to a file
        with open("analysis.txt", "w", encoding="utf-8") as f:
            f.write(analysis_result)
        print("Analysis saved to analysis.txt")
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")


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

    def fly_forward_and_capture_frames(self):
        """
        Flies the drone forward for 10 seconds and captures one frame per second.
        The captured frames are stored internally for later analysis.
        """
        self.captured_frames = fly_forward_and_capture(self.drone, flight_duration=7)
        return f"Successfully captured {len(self.captured_frames)} frames."

    def rotate_90_degrees(self):
        """Rotates the drone 90 degrees clockwise."""
        print("Rotating 90 degrees...")
        rotation_duration = 1.5  # seconds, adjusted from 2.0 to fix 120->90 degree issue
        command_interval = 0.02  # seconds
        num_iterations = int(rotation_duration / command_interval)

        for _ in range(num_iterations):
            self.drone.move(pitch=0, yaw=1.0, roll=0, ascent=0)
            time.sleep(command_interval)
        
        # Stop rotation
        self.drone.move(0, 0, 0, 0)
        print("Rotation complete.")
        return "Rotated 90 degrees."

    def analyze_captured_frames(self):
        """
        Analyzes the frames captured during the flight using OpenAI.
        This tool must be used after flying and capturing frames.
        """
        if not self.captured_frames:
            return "No frames have been captured yet. Use fly_forward_and_capture_frames first."
        
        analyze_frames_with_openai(
            self.captured_frames, 
            self.openai_client, 
            10
        )
        return "Analysis is complete. The result has been saved to analysis.txt."


def run_test_flight_pattern(controller: DroneAgentController):
    """Runs a pre-defined flight pattern for testing."""
    print("Running test flight pattern...")
    print("1. Flying forward.")
    controller.fly_forward_and_capture_frames()
    print("2. Flying forward again.")
    controller.fly_forward_and_capture_frames()
    print("3. Rotating 90 degrees.")
    controller.rotate_90_degrees()
    print("4. Flying forward.")
    controller.fly_forward_and_capture_frames()
    print("5. Rotating 90 degrees.")
    controller.rotate_90_degrees()
    print("Test flight pattern complete.")


# Connect to the drone
with OpenDJI(IP_ADDR) as drone:
    if openai_client is None:
        print("OpenAI client not initialized. Exiting application.")
    else:
        controller = DroneAgentController(drone, openai_client)

        # Takeoff before executing agent tasks
        controller.takeoff()

        # @tool
        # def fly_forward_and_capture_frames():
        #     """
        #     Flies the drone forward for 10 seconds and captures one frame per second.
        #     The captured frames are stored internally for later analysis.
        #     """
        #     return controller.fly_forward_and_capture_frames()

        # @tool
        # def rotate_90_degrees():
        #     """Rotates the drone 90 degrees clockwise."""
        #     return controller.rotate_90_degrees()

        # @tool
        # def analyze_captured_frames():
        #     """
        #     Analyzes the frames captured during the flight using OpenAI.
        #     This tool must be used after flying and capturing frames.
        #     """
        #     return controller.analyze_captured_frames()

        # tools = [
        #     fly_forward_and_capture_frames,
        #     rotate_90_degrees,
        #     analyze_captured_frames,
        # ]

        # agent = CodeAgent(
        #     tools=tools,
        #     model="gpt-4o",
        # )

        # prompt = "You are a drone operations agent. Your goal is to safely fly a drone to collect visual data and then analyze it. The drone is already flying. Your task is to fly forward once, rotate 90 degrees, and then analyze the captured frames."
        # print(f"Agent executing prompt: '{prompt}'")
        # agent.run(prompt)
        
        run_test_flight_pattern(controller)
        
        # Land after executing agent tasks
        controller.land()


