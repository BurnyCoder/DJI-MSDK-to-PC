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
import cv2 # Added for image processing with YOLO
from ultralytics import YOLO # Added for YOLO object detection
import json # Added for handling YOLO model info (potentially)
import uuid # Added for generating unique log file names

# Import the new YOLO tracker function
from yolo_tracker_old import track_person_and_rotate_yolo

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
# --- Global YOLO Model ---
yolo_model = None

# --- Logging Function ---
def log_message(log_file_name: str, message: str):
    """Logs a message to a file and prints it to the console."""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    file_path = os.path.join(logs_dir, log_file_name)
    
    # Get current timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    formatted_message = f"[{timestamp}] {message}"
    
    with open(file_path, "a") as f:
        f.write(formatted_message + "\n")
    print(formatted_message)

def initialize_drone_connection():
    """Initializes the global drone connection."""
    global drone_connection
    if drone_connection is None:
        try:
            log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Attempting to connect to drone at IP: {DRONE_IP_ADDR}...")
            drone_connection = OpenDJI(DRONE_IP_ADDR)
            log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Successfully connected to the drone.")
            # Optionally, enable control right after connection
            # result = drone_connection.enableControl(True)
            # print(f"Enable SDK control attempt post-connection. Drone response: {result}")
        except Exception as e:
            log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Failed to connect to the drone: {e}")
            drone_connection = None # Ensure it's None on failure
            # raise ConnectionError(f"Failed to initialize drone connection: {e}") # Or raise an error
    return drone_connection

def close_drone_connection():
    """Closes the global drone connection if it's open."""
    global drone_connection
    if drone_connection:
        try:
            log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Closing drone connection...")
            # result = drone_connection.disableControl(True) # Optionally disable control before closing
            # print(f"Disable SDK control attempt pre-close. Drone response: {result}")
            drone_connection.close()
            log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Drone connection closed.")
        except Exception as e:
            log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Error closing drone connection: {e}")
        finally:
            drone_connection = None

# Register the close function to be called on exit
atexit.register(close_drone_connection)

# --- YOLO Model Initialization ---
def initialize_yolo_model():
    """Initializes the global YOLO model."""
    global yolo_model
    if yolo_model is None:
        log_message(f"yolo_model_log_{uuid.uuid4().hex[:8]}.txt", "Starting YOLO model initialization...")
        try:
            log_message(f"yolo_model_log_{uuid.uuid4().hex[:8]}.txt", "Attempting to load YOLO model (yolov8n.pt)...")
            start_time = time.time()
            # Ensure 'yolov8n.pt' is accessible in the environment where this script runs
            yolo_model = YOLO("yolov8n.pt")
            load_time = time.time() - start_time
            log_message(f"yolo_model_log_{uuid.uuid4().hex[:8]}.txt", f"YOLO model loaded successfully in {load_time:.2f} seconds.")
            # Optional: Log model details if needed
            # model_info = {"model_type": "yolov8n.pt", "task": yolo_model.task, "device": str(yolo_model.device)}
            # print(f"YOLO model info: {json.dumps(model_info)}")
        except Exception as e:
            log_message(f"yolo_model_log_{uuid.uuid4().hex[:8]}.txt", f"Error loading YOLO model: {e}")
            log_message(f"yolo_model_log_{uuid.uuid4().hex[:8]}.txt", "Please ensure the YOLO model file (e.g., 'yolov8n.pt') is available.")
            yolo_model = None # Ensure it's None on failure
    return yolo_model

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
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Enable SDK control command sent. Drone response: {result}")
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
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Enable SDK control command sent. Drone response: {result}")
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
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Enable SDK control command sent. Drone response: {result}")
        drone_connection.move(rcw, du, lr, bf)
        return f"Move command sent: rcw={rcw}, du={du}, lr={lr}, bf={bf}"
    except Exception as e:
        return f"Error moving drone: {str(e)}"

@tool
def move_forward_one_meter() -> str:
    """
    Commands the drone to move forward approximately one meter.

    Note: The actual distance is an estimate based on a fixed duration and speed.
    It might vary depending on drone model, battery, and environmental conditions.

    Returns:
        str: A message indicating the result of the command.
    """
    global drone_connection
    if drone_connection is None:
        initialize_drone_connection()
        if drone_connection is None:
            return "Error: Drone connection not established. Cannot move forward."

    FORWARD_SPEED = 1 # Speed value between 0.0 and 1.0
    FORWARD_DURATION = 3 # Estimated duration in seconds to cover 1 meter at FORWARD_SPEED

    try:
        result = drone_connection.enableControl(True)
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Enable SDK control command sent. Drone response: {result}")
        # --- Takeoff --- 
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Sending takeoff command...")
        takeoff_result = drone_connection.takeoff(True)
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Takeoff result: {takeoff_result}")
        if "error" in str(takeoff_result).lower() or "failed" in str(takeoff_result).lower():
            return f"Takeoff failed, cannot start tracking: {takeoff_result}"
        
        # Give a brief moment for the drone to stabilize after takeoff
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Stabilizing after takeoff...")
        time.sleep(10)

        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Attempting to move forward for {FORWARD_DURATION}s at speed {FORWARD_SPEED}...")

        # Start moving forward
        drone_connection.move(rcw=0.0, du=0.0, lr=0.0, bf=FORWARD_SPEED)
        time.sleep(FORWARD_DURATION)

        # Stop moving
        drone_connection.move(rcw=0.0, du=0.0, lr=0.0, bf=0.0)
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Movement stopped.")

        time.sleep(1)

        drone_connection.land(True)

        # Optionally disable control if preferred after each discrete action
        # disable_result = drone_connection.disableControl(True)
        # print(f"Disable SDK control command sent. Drone response: {disable_result}")

        return f"Move forward command executed for {FORWARD_DURATION} seconds."
    except Exception as e:
        # Ensure movement stops in case of error during sleep
        try:
            drone_connection.move(rcw=0.0, du=0.0, lr=0.0, bf=0.0)
        except Exception as stop_e:
            log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Error stopping drone after move error: {stop_e}")
        return f"Error moving forward: {str(e)}"

@tool
def rotate_90_degrees_clockwise() -> str:
    """
    Commands the drone to rotate approximately 90 degrees clockwise.

    Note: The actual angle is an estimate based on a fixed duration and rotation speed.
    It might vary depending on drone model, battery, and environmental conditions.

    Returns:
        str: A message indicating the result of the command.
    """
    global drone_connection
    if drone_connection is None:
        initialize_drone_connection()
        if drone_connection is None:
            return "Error: Drone connection not established. Cannot rotate."

    ROTATION_SPEED = 1 # Rotation speed value between 0.0 and 1.0
    ROTATION_DURATION = 1.5 # Estimated duration in seconds for 90 degrees at ROTATION_SPEED

    try:
        result = drone_connection.enableControl(True)
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Enable SDK control command sent. Drone response: {result}")
        # --- Takeoff --- 
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Sending takeoff command...")
        takeoff_result = drone_connection.takeoff(True)
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Takeoff result: {takeoff_result}")
        if "error" in str(takeoff_result).lower() or "failed" in str(takeoff_result).lower():
            return f"Takeoff failed, cannot start tracking: {takeoff_result}"
        
        # Give a brief moment for the drone to stabilize after takeoff
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Stabilizing after takeoff...")
        time.sleep(10)

        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Attempting to rotate clockwise for {ROTATION_DURATION}s at speed {ROTATION_SPEED}...")

        # Start rotating
        drone_connection.move(rcw=-ROTATION_SPEED, du=0.0, lr=0.0, bf=0.0)
        time.sleep(ROTATION_DURATION)

        # Stop rotating
        drone_connection.move(rcw=0.0, du=0.0, lr=0.0, bf=0.0)
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Rotation stopped.")

        time.sleep(1)

        drone_connection.land(True)

        # Optionally disable control
        # disable_result = drone_connection.disableControl(True)
        # print(f"Disable SDK control command sent. Drone response: {disable_result}")

        return f"Rotate clockwise command executed for {ROTATION_DURATION} seconds."
    except Exception as e:
        # Ensure rotation stops in case of error during sleep
        try:
            drone_connection.move(rcw=0.0, du=0.0, lr=0.0, bf=0.0)
        except Exception as stop_e:
            log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Error stopping drone after rotate error: {stop_e}")
        return f"Error rotating clockwise: {str(e)}"

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
            log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "No frame available from the drone.")
            # Maintaining the string return type for error, as per original function's behavior.
            raise ValueError("No frame available from the drone.") 

        pil_image = Image.fromarray(frame_np.astype(np.uint8)) # Ensure correct dtype for PIL
        return AgentImage(pil_image)
    except ValueError as ve: # Catch the specific error for no frame
        return f"Error getting drone frame: {str(ve)}" 
    except Exception as e:
        return f"Error getting drone frame info: {str(e)}"

# --- YOLO Analysis Function (Adapted from ai_processing.py) ---
def analyze_image_with_yolo(image_frame, log_file_name: str):
    """Analyzes an image frame using YOLO and returns results."""
    global yolo_model
    if yolo_model is None:
        initialize_yolo_model()
        if yolo_model is None:
            # Log this specific failure via the passed-in logger if available
            log_message(log_file_name, "Error: YOLO model not initialized within analyze_image_with_yolo.")
            return None, "Error: YOLO model not initialized."

    log_message(log_file_name, "Starting YOLO object detection")
    try:
        # Log image information for debugging
        height, width, channels = image_frame.shape
        log_message(log_file_name, f"Input image for YOLO: {width}x{height}x{channels}")

        # Run YOLO inference
        start_time = time.time()
        results = yolo_model(image_frame)
        detection_time = time.time() - start_time
        log_message(log_file_name, f"YOLO detection completed in {detection_time:.2f} seconds")

        if results and results[0].boxes:
            # Count detected objects
            total_objects = len(results[0].boxes)
            log_message(log_file_name, f"YOLO detected {total_objects} objects")

            # Log object classes and confidence
            detected_classes = {}
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = yolo_model.names[class_id]
                confidence = float(box.conf[0])

                if class_name in detected_classes:
                    detected_classes[class_name].append(confidence)
                else:
                    detected_classes[class_name] = [confidence]

            detection_summary = []
            for class_name, confidences in detected_classes.items():
                avg_conf = sum(confidences) / len(confidences)
                summary = f"Detected {len(confidences)}x {class_name} (avg conf: {avg_conf:.2f})"
                log_message(log_file_name, summary) # Log each summary line
                detection_summary.append(summary)

            return results, "\n".join(detection_summary)
        else:
            log_message(log_file_name, "No objects detected by YOLO")
            return None, "No objects detected by YOLO."
    except Exception as e:
        log_message(log_file_name, f"Error during YOLO analysis: {e}")
        # Consider logging traceback here if needed: import traceback; traceback.print_exc()
        return None, f"Error during YOLO analysis: {e}"

@tool
def analyze_frame_with_yolo() -> str:
    """
    Retrieves the current video frame from the drone and analyzes it using YOLOv8.

    Returns:
        str: A summary of detected objects or an error message.
    """
    global drone_connection
    global yolo_model
    # For standalone tool calls, generate a unique log file name.
    tool_log_file_name = f"analyze_frame_yolo_tool_log_{uuid.uuid4().hex[:8]}.txt"

    # Ensure drone is connected
    if drone_connection is None:
        initialize_drone_connection() # Uses its own print statements, which is fine
        if drone_connection is None:
            # Log this specific failure for the tool's log
            log_message(tool_log_file_name, "Error: Drone connection not established for analyze_frame_with_yolo.")
            return "Error: Drone connection not established. Cannot get frame for YOLO analysis."

    # Ensure YOLO model is loaded
    if yolo_model is None:
        initialize_yolo_model() # Uses its own print statements, which is fine
        if yolo_model is None:
            log_message(tool_log_file_name, "Error: YOLO model failed to initialize for analyze_frame_with_yolo.")
            return "Error: YOLO model failed to initialize. Cannot analyze frame."

    try:
        log_message(tool_log_file_name, "Attempting to get frame for YOLO analysis...")
        frame_np = drone_connection.getFrame()
        if frame_np is None:
            log_message(tool_log_file_name, "Error: No frame available from the drone for YOLO analysis.")
            return "Error: No frame available from the drone for YOLO analysis."

        # Analyze the frame using the dedicated YOLO function
        # When called as a tool, analyze_image_with_yolo will use the log file name passed here.
        _yolo_results_obj, yolo_summary = analyze_image_with_yolo(frame_np, tool_log_file_name)
        # We return the summary string, not the full results object
        return yolo_summary

    except Exception as e:
        log_message(tool_log_file_name, f"Error during YOLO frame analysis: {str(e)}")
        return f"Error during YOLO frame analysis: {str(e)}"

@tool
def track_person_and_rotate_llm(max_iterations: int = 30, seconds_per_iteration: float = 1) -> str:
    """
    Commands the drone to take off, then continuously uses OpenAI's vision model
    to analyze the video feed and determine appropriate movements (rotation,
    forward/backward, up/down, left/right) and their duration to track a person.
    If no person is detected, it asks the model for scanning or holding maneuvers.
    Finally, commands the drone to land.

    Args:
        max_iterations: The maximum number of tracking attempts.
        seconds_per_iteration: The target total cycle time for each iteration (includes processing, potential movement, and waiting). Minimum time between analyses.

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
        time.sleep(10)

        print(f"Starting person tracking for up to {max_iterations} iterations.")
        person_sighted_in_previous_iteration = False
        consecutive_no_person_scans = 0 # Tracks how many consecutive frames a person isn't seen after being seen

        for i in range(max_iterations):
            print(f"Tracking iteration {i+1}/{max_iterations}...")
            # Initialize movement parameters for this iteration
            current_rcw, current_du, current_lr, current_bf, current_duration = 0.0, 0.0, 0.0, 0.0, 0.0
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

                # Determine context for the LLM
                if person_sighted_in_previous_iteration:
                    state_context = "A person was visible in the last frame. Continue tracking."
                elif consecutive_no_person_scans > 0:
                    state_context = f"A person was visible previously but has been lost for {consecutive_no_person_scans} frame(s). Decide whether to scan or hold position."
                else:
                    state_context = "No person has been sighted recently. Scan for a person or hold position."

                # Using OpenAI for image analysis and movement decisions
                print(f"Sending frame to OpenAI for analysis ({state_context})...")
                prompt = f"""Analyze this image from a drone's camera. Context: {state_context}
Instructions:
1. Determine if a person is clearly visible (Yes/No).
2. Provide drone movement commands (`rotation` for rotation, `move` for forward/backward) and `duration` (seconds).
   - If Visible: Yes, provide `rotation` to turn towards the person and `move` (positive value, 0.0 to 1.0) to move forward.
   - If Visible: No, provide `rotation` (non-zero) to scan for the person. Keep `move` at 0.0.
   - `rotation` range: -1.0 (rotate left) to 1.0 (rotate right).
   - `move` range: 0.0 to 1.0 (only forward movement considered).
   - `duration` range: 0.0 to 5.0 seconds (how long to apply the movement).
Response Format:
Respond ONLY in the format: "Visible: [Yes/No], rotation: [float], move: [float], duration: [float]"

Examples:
- Person centered, move forward (1 meter): "Visible: Yes, rotation: 0.0, move: 1.0, duration: 3.0"
- Person slightly right, move forward: "Visible: Yes, rotation: 0.1, move: 1.0, duration: 3.0"
- No person, scan right (approx 90 degrees clockwise): "Visible: No, rotation: -1.0, move: 0.0, duration: 1.5"
- No person, scan left (approx 90 degrees anti-clockwise): "Visible: No, rotation: 1.0, move: 0.0, duration: 1.5" """

                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                            ]
                        }
                    ],
                )

                llm_output = response.choices[0].message.content.strip()
                print(f"OpenAI analysis result: '{llm_output}'")

                # Parse the structured response
                is_visible = False
                try:
                    # Example: "Visible: Yes, rotation: 0.15, move: 0.1, duration: 1.5"
                    parts = llm_output.split(',')
                    visible_part = parts[0].split(':')[1].strip().lower()
                    rotation_part = parts[1].split(':')[1].strip()
                    move_part = parts[2].split(':')[1].strip()
                    duration_part = parts[3].split(':')[1].strip()

                    if visible_part == 'yes':
                        is_visible = True
                    
                    current_rcw = max(-1.0, min(1.0, float(rotation_part))) # Assign rotation to rcw
                    current_bf = max(0.0, min(1.0, float(move_part)))     # Assign move to bf
                    current_duration = max(0.0, float(duration_part))     # Ensure duration is non-negative
                    
                    # Hardcode du and lr to 0.0
                    current_du = 0.0
                    current_lr = 0.0

                except Exception as parse_error:
                    print(f"Warning: Could not parse OpenAI response '{llm_output}'. Error: {parse_error}. Holding position.")
                    is_visible = False
                    current_rcw, current_du, current_lr, current_bf, current_duration = 0.0, 0.0, 0.0, 0.0, 0.0

                # Update state for next iteration's context
                if is_visible:
                    print(f"Person detected. LLM suggests move: rotation={current_rcw:.2f}, move={current_bf:.2f}, duration={current_duration:.2f}s (vertical/sideways hardcoded to 0.0)")
                    person_sighted_in_previous_iteration = True
                    consecutive_no_person_scans = 0
                else:
                    if person_sighted_in_previous_iteration:
                         print("Person lost.")
                         consecutive_no_person_scans = 1
                    elif consecutive_no_person_scans > 0:
                         print(f"Person still not found (lost for {consecutive_no_person_scans + 1} frames).")
                         consecutive_no_person_scans += 1
                    else:
                         print("No person detected.")
                         
                    person_sighted_in_previous_iteration = False
                    
                    if current_duration > 0:
                        print(f"LLM suggests action (scan/hold): rotation={current_rcw:.2f}, move={current_bf:.2f}, duration={current_duration:.2f}s (vertical/sideways hardcoded to 0.0)")
                    else:
                        print("LLM suggests holding position.")


                # Apply movement if duration is positive
                if current_duration > 0:
                    print(f"Executing LLM-defined movement for {current_duration:.2f}s...")
                    drone.move(current_rcw, current_du, current_lr, current_bf) # Apply LLM command
                    time.sleep(current_duration) # Move for the LLM-specified duration
                    print("Stopping movement.")
                    drone.move(0, 0, 0, 0) # Stop all movement
                else:
                    print("No movement adjustment needed for this iteration (duration is 0).")

            except Exception as e:
                print(f"Error in tracking iteration {i+1}: {str(e)}")
            
            iteration_logic_end_time = time.time()
            time_spent_in_iteration_logic = iteration_logic_end_time - iteration_logic_start_time
            
            remaining_wait_time = seconds_per_iteration - time_spent_in_iteration_logic
            
            if remaining_wait_time > 0:
                print(f"Waiting for {remaining_wait_time:.2f} seconds before next iteration's processing...")
                #time.sleep(remaining_wait_time)
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

        # Ensure YOLO model is initialized when agent is created
        initialize_yolo_model()
        if yolo_model is None:
            print("WARNING: YOLO model failed to initialize for AutonomousDroneAgent. YOLO analysis will not be available.")

        # Create the ToolCallingAgent with the defined drone tools
        # The tools are now defined outside the class
        self.drone_agent = ToolCallingAgent(
            tools=[
                drone_takeoff,
                drone_land,
                move_drone,
                move_forward_one_meter,
                rotate_90_degrees_clockwise,
                get_drone_frame_info,
                analyze_frame_with_yolo,
                track_person_and_rotate_llm,
                track_person_and_rotate_yolo,
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
            
            # Uncomment one of these to test (requires drone connection)
            print("\n--- Running Example ---")

            # Example: Move forward and rotate
            # print("\n--- Running Example: Move Forward ---")
            # move_forward_one_meter()

            # print("\n--- Running Example: Rotate ---")
            #rotate_90_degrees_clockwise()

            # Option 2 example:
            # print("\n--- Running Example: Person Tracking ---")
            # result = track_person_and_rotate_llm(max_iterations=100000)
            # print(result)

            # Option 3 example:
            # print("\n--- Running Example: Query Agent ---")
            # response = agent_instance.run_query("Take off the drone and find a person.")
            # print(response)

            # Option 4 example: YOLO Tracking
            print("\n--- Running Example: YOLO Person Tracking ---")
            response = agent_instance.run_query("Use YOLO to track a person.")
            print(response)

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
