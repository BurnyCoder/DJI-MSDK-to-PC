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
# DRONE_IP_ADDR = os.getenv("DRONE_IP_ADDR", "192.168.1.115") # Removed
# OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- Global Drone Connection ---
# drone_connection = None # Removed
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

def initialize_drone_connection(ip_address: str) -> OpenDJI | None:
    """Initializes a drone connection to the given IP address.
    
    Args:
        ip_address: The IP address of the drone.
        
    Returns:
        An OpenDJI instance if connection is successful, None otherwise.
    """
    try:
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Attempting to connect to drone at IP: {ip_address}...")
        connection = OpenDJI(ip_address)
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Successfully connected to the drone at {ip_address}.")
        return connection
    except Exception as e:
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Failed to connect to the drone at {ip_address}: {e}")
        return None

def close_drone_connection(drone_instance: OpenDJI):
    """Closes the given drone connection if it's open."""
    if drone_instance:
        try:
            log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Closing drone connection for {drone_instance.host_address}...")
            drone_instance.close()
            log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Drone connection for {drone_instance.host_address} closed.")
        except Exception as e:
            log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Error closing drone connection for {drone_instance.host_address}: {e}")

# Register the close function to be called on exit
# atexit.register(close_drone_connection) # Removed due to refactoring for multiple drone instances

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
def _tool_drone_takeoff(drone_instance: OpenDJI) -> str:
    """
    Commands the drone to take off.

    Args:
        drone_instance: The OpenDJI instance for the connected drone.

    Returns:
        str: A message indicating the result of the takeoff command.
    """
    # global drone_connection # Removed
    if drone_instance is None:
        # initialize_drone_connection() # Removed
        # if drone_connection is None: # Removed
        return "Error: Drone instance not provided or not connected. Cannot take off."
    try:
        result = drone_instance.enableControl(True)
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Enable SDK control command sent. Drone response: {result}")
        result = drone_instance.takeoff(True)
        return f"Takeoff command sent. Drone response: {result}"
    except Exception as e:
        return f"Error during takeoff: {str(e)}"

@tool
def _tool_drone_land(drone_instance: OpenDJI) -> str:
    """
    Commands the drone to land.

    Args:
        drone_instance: The OpenDJI instance for the connected drone.

    Returns:
        str: A message indicating the result of the land command.
    """
    # global drone_connection # Removed
    if drone_instance is None:
        # initialize_drone_connection() # Removed
        # if drone_connection is None: # Removed
        return "Error: Drone instance not provided or not connected. Cannot land."
    try:
        result = drone_instance.enableControl(True)
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Enable SDK control command sent. Drone response: {result}")
        result = drone_instance.land(True)
        return f"Land command sent. Drone response: {result}"
    except Exception as e:
        return f"Error during land: {str(e)}"

@tool
def _tool_move_drone(drone_instance: OpenDJI, rcw: float, du: float, lr: float, bf: float) -> str:
    """
    Moves the drone with specified control values.

    Args:
        drone_instance: The OpenDJI instance for the connected drone.
        rcw: Rotational movement (rotate clockwise/anti-clockwise).
        du: Vertical movement (down/up).
        lr: Sideways movement (left/right).
        bf: Forward/backward movement.

    Returns:
        str: A message indicating the result of the move command.
    """
    # global drone_connection # Removed
    if drone_instance is None:
        # initialize_drone_connection() # Removed
        # if drone_connection is None: # Removed
        return "Error: Drone instance not provided or not connected. Cannot move."
    try:
        result = drone_instance.enableControl(True)
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Enable SDK control command sent. Drone response: {result}")
        drone_instance.move(rcw, du, lr, bf)
        return f"Move command sent: rcw={rcw}, du={du}, lr={lr}, bf={bf}"
    except Exception as e:
        return f"Error moving drone: {str(e)}"

@tool
def _tool_move_forward_one_meter(drone_instance: OpenDJI) -> str:
    """
    Commands the drone to move forward approximately one meter.

    Args:
        drone_instance: The OpenDJI instance for the connected drone.

    Returns:
        str: A message indicating the result of the command.
    """
    # global drone_connection # Removed
    if drone_instance is None:
        # initialize_drone_connection() # Removed
        # if drone_connection is None: # Removed
        return "Error: Drone instance not provided or not connected. Cannot move forward."

    FORWARD_SPEED = 1 # Speed value between 0.0 and 1.0
    FORWARD_DURATION = 3 # Estimated duration in seconds to cover 1 meter at FORWARD_SPEED

    try:
        result = drone_instance.enableControl(True)
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Enable SDK control command sent. Drone response: {result}")
        # --- Takeoff --- 
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Sending takeoff command...")
        takeoff_result = drone_instance.takeoff(True)
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Takeoff result: {takeoff_result}")
        if "error" in str(takeoff_result).lower() or "failed" in str(takeoff_result).lower():
            return f"Takeoff failed, cannot start tracking: {takeoff_result}"
        
        # Give a brief moment for the drone to stabilize after takeoff
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Stabilizing after takeoff...")
        time.sleep(10)

        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Attempting to move forward for {FORWARD_DURATION}s at speed {FORWARD_SPEED}...")

        # Start moving forward
        drone_instance.move(rcw=0.0, du=0.0, lr=0.0, bf=FORWARD_SPEED)
        time.sleep(FORWARD_DURATION)

        # Stop moving
        drone_instance.move(rcw=0.0, du=0.0, lr=0.0, bf=0.0)
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Movement stopped.")

        time.sleep(1)

        drone_instance.land(True)

        return f"Move forward command executed for {FORWARD_DURATION} seconds."
    except Exception as e:
        try:
            drone_instance.move(rcw=0.0, du=0.0, lr=0.0, bf=0.0)
        except Exception as stop_e:
            log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Error stopping drone after move error: {stop_e}")
        return f"Error moving forward: {str(e)}"

@tool
def _tool_rotate_90_degrees_clockwise(drone_instance: OpenDJI) -> str:
    """
    Commands the drone to rotate approximately 90 degrees clockwise.

    Args:
        drone_instance: The OpenDJI instance for the connected drone.

    Returns:
        str: A message indicating the result of the command.
    """
    # global drone_connection # Removed
    if drone_instance is None:
        # initialize_drone_connection() # Removed
        # if drone_connection is None: # Removed
        return "Error: Drone instance not provided or not connected. Cannot rotate."

    ROTATION_SPEED = 1 # Rotation speed value between 0.0 and 1.0
    ROTATION_DURATION = 1.5 # Estimated duration in seconds for 90 degrees at ROTATION_SPEED

    try:
        result = drone_instance.enableControl(True)
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Enable SDK control command sent. Drone response: {result}")
        # --- Takeoff --- 
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Sending takeoff command...")
        takeoff_result = drone_instance.takeoff(True)
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Takeoff result: {takeoff_result}")
        if "error" in str(takeoff_result).lower() or "failed" in str(takeoff_result).lower():
            return f"Takeoff failed, cannot start tracking: {takeoff_result}"
        
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Stabilizing after takeoff...")
        time.sleep(10)

        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Attempting to rotate clockwise for {ROTATION_DURATION}s at speed {ROTATION_SPEED}...")

        drone_instance.move(rcw=-ROTATION_SPEED, du=0.0, lr=0.0, bf=0.0)
        time.sleep(ROTATION_DURATION)

        drone_instance.move(rcw=0.0, du=0.0, lr=0.0, bf=0.0)
        log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Rotation stopped.")

        time.sleep(1)

        drone_instance.land(True)

        return f"Rotate clockwise command executed for {ROTATION_DURATION} seconds."
    except Exception as e:
        try:
            drone_instance.move(rcw=0.0, du=0.0, lr=0.0, bf=0.0)
        except Exception as stop_e:
            log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Error stopping drone after rotate error: {stop_e}")
        return f"Error rotating clockwise: {str(e)}"

@tool
def _tool_get_drone_frame_info(drone_instance: OpenDJI) -> AgentImage | str:
    """
    Retrieves the current video frame from the drone as an AgentImage.

    Args:
        drone_instance: The OpenDJI instance for the connected drone.

    Returns:
        AgentImage: An AgentImage object containing the frame, or a string with an error message if unsuccessful.
    """
    # global drone_connection # Removed
    if drone_instance is None:
        # initialize_drone_connection() # Removed
        # if drone_connection is None: # Removed
        return "Error: Drone instance not provided or not connected. Cannot get frame."
    try:
        frame_np = drone_instance.getFrame()
        if frame_np is None:
            log_message(f"drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "No frame available from the drone.")
            raise ValueError("No frame available from the drone.") 

        pil_image = Image.fromarray(frame_np.astype(np.uint8))
        return AgentImage(pil_image)
    except ValueError as ve:
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
def _tool_analyze_frame_with_yolo(drone_instance: OpenDJI) -> str:
    """
    Retrieves the current video frame from the drone and analyzes it using YOLOv8.

    Args:
        drone_instance: The OpenDJI instance for the connected drone.

    Returns:
        str: A summary of detected objects or an error message.
    """
    # global drone_connection # Removed
    # global yolo_model # yolo_model is still global and initialized by initialize_yolo_model()
    tool_log_file_name = f"analyze_frame_yolo_tool_log_{uuid.uuid4().hex[:8]}.txt"

    if drone_instance is None:
        # initialize_drone_connection() # Removed
        # if drone_connection is None: # Removed
        log_message(tool_log_file_name, "Error: Drone instance not provided for analyze_frame_with_yolo.")
        return "Error: Drone instance not provided. Cannot get frame for YOLO analysis."

    global yolo_model # yolo_model is global and initialized by initialize_yolo_model()
    if yolo_model is None:
        initialize_yolo_model()
        if yolo_model is None:
            log_message(tool_log_file_name, "Error: YOLO model failed to initialize for analyze_frame_with_yolo.")
            return "Error: YOLO model failed to initialize. Cannot analyze frame."

    try:
        log_message(tool_log_file_name, "Attempting to get frame for YOLO analysis...")
        frame_np = drone_instance.getFrame()
        if frame_np is None:
            log_message(tool_log_file_name, "Error: No frame available from the drone for YOLO analysis.")
            return "Error: No frame available from the drone for YOLO analysis."

        _yolo_results_obj, yolo_summary = analyze_image_with_yolo(frame_np, tool_log_file_name)
        return yolo_summary

    except Exception as e:
        log_message(tool_log_file_name, f"Error during YOLO frame analysis: {str(e)}")
        return f"Error during YOLO frame analysis: {str(e)}"

@tool
def _tool_track_person_and_rotate_llm(drone_instance: OpenDJI, max_iterations: int = 30, seconds_per_iteration: float = 1) -> str:
    """
    Commands the drone to take off, then continuously uses OpenAI's vision model
    to analyze the video feed and determine appropriate movements (rotation,
    forward/backward, up/down, left/right) and their duration to track a person.
    If no person is detected, it asks the model for scanning or holding maneuvers.
    Finally, commands the drone to land.

    Args:
        drone_instance: The OpenDJI instance for the connected drone.
        max_iterations: The maximum number of tracking attempts.
        seconds_per_iteration: The target total cycle time for each iteration (includes processing, potential movement, and waiting). Minimum time between analyses.

    Returns:
        str: A message indicating the result of the tracking sequence.
    """
    print("Initiating automated person tracking sequence...")
    # global drone_connection # Removed
    
    if drone_instance is None:
        # initialize_drone_connection() # Removed
        # if drone_connection is None: # Removed
        return "Error: Drone instance not provided or not connected. Cannot start tracking."

    try:
        # Use the provided drone_instance
        drone = drone_instance 
        
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
        self.model = LiteLLMModel(
            model_id="openrouter/google/gemini-2.5-pro-exp-03-25", # Example model
            temperature=0.5,
            max_tokens=50000 # Increased token limit
        )

        self.drone_instances: dict[str, OpenDJI] = {}
        self.active_drone_ip: str | None = None
        
        # Ensure YOLO model is initialized when agent is created (if still needed globally)
        # Depending on usage, YOLO model could also be managed per drone or on demand.
        initialize_yolo_model() 
        if yolo_model is None:
            print("WARNING: Global YOLO model failed to initialize for AutonomousDroneAgent. YOLO analysis might not be available or might need drone-specific initialization.")

        # Register the disconnect_all_drones method to be called on exit
        atexit.register(self.disconnect_all_drones)

        # Define wrapper methods for tools within the agent
        # These wrappers will be passed to ToolCallingAgent

        @tool
        def drone_takeoff() -> str:
            drone = self.get_active_drone_instance()
            if not drone:
                return "Error: No active drone selected or connected. Cannot take off."
            return _tool_drone_takeoff(drone)

        @tool
        def drone_land() -> str:
            drone = self.get_active_drone_instance()
            if not drone:
                return "Error: No active drone selected or connected. Cannot land."
            return _tool_drone_land(drone)

        @tool
        def move_drone(rcw: float, du: float, lr: float, bf: float) -> str:
            drone = self.get_active_drone_instance()
            if not drone:
                return "Error: No active drone selected or connected. Cannot move."
            return _tool_move_drone(drone, rcw, du, lr, bf)

        @tool
        def move_forward_one_meter() -> str:
            drone = self.get_active_drone_instance()
            if not drone:
                return "Error: No active drone selected or connected. Cannot move forward."
            return _tool_move_forward_one_meter(drone)

        @tool
        def rotate_90_degrees_clockwise() -> str:
            drone = self.get_active_drone_instance()
            if not drone:
                return "Error: No active drone selected or connected. Cannot rotate."
            return _tool_rotate_90_degrees_clockwise(drone)

        @tool
        def get_drone_frame_info() -> AgentImage | str: # Ensure return type matches underlying tool
            drone = self.get_active_drone_instance()
            if not drone:
                return "Error: No active drone selected or connected. Cannot get frame."
            return _tool_get_drone_frame_info(drone)

        @tool
        def analyze_frame_with_yolo() -> str:
            drone = self.get_active_drone_instance()
            if not drone:
                return "Error: No active drone selected or connected. Cannot analyze frame."
            # _tool_analyze_frame_with_yolo itself handles yolo model initialization check
            return _tool_analyze_frame_with_yolo(drone)

        @tool
        def track_person_and_rotate_llm(max_iterations: int = 30, seconds_per_iteration: float = 1) -> str:
            drone = self.get_active_drone_instance()
            if not drone:
                return "Error: No active drone selected or connected. Cannot track person (LLM)."
            return _tool_track_person_and_rotate_llm(drone, max_iterations, seconds_per_iteration)
        
        # Placeholder for the imported yolo_tracker tool
        # This assumes yolo_tracker_old.track_person_and_rotate_yolo will be refactored
        # to accept a drone_instance as its first argument.
        @tool
        def track_person_and_rotate_yolo_wrapper(max_frames: int = 100, target_name: str = "person", confidence_threshold: float = 0.6, log_to_file: bool = True) -> str:
            drone = self.get_active_drone_instance()
            if not drone:
                return "Error: No active drone selected or connected. Cannot track with YOLO."
            try:
                # Dynamically import here if preferred, or ensure it's imported at the top
                from yolo_tracker_old import track_person_and_rotate_yolo as yolo_tracker_tool
                # The original tool from yolo_tracker_old.py needs to be refactored
                # to accept drone_instance as the first argument.
                # For now, this is a placeholder call.
                # return yolo_tracker_tool(drone, max_frames, target_name, confidence_threshold, log_to_file)
                return f"YOLO tracking (external tool) would be called for drone {self.active_drone_ip}. (Needs refactor in yolo_tracker_old.py)"
            except ImportError:
                return "Error: yolo_tracker_old.py or its track_person_and_rotate_yolo tool not found or not refactored."
            except TypeError as te: # Catch if the underlying tool doesn't accept drone_instance yet
                 return f"Error calling yolo_tracker_tool: {te}. It might not be refactored to accept a drone instance."


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
                track_person_and_rotate_yolo_wrapper, # Use the wrapper
            ],
            model=self.model
        )

    def connect_to_drone(self, ip_address: str, make_active: bool = True) -> bool:
        """Connects to a drone at the given IP address."""
        if ip_address in self.drone_instances:
            print(f"Already connected to drone at {ip_address}.")
            if make_active:
                self.active_drone_ip = ip_address
                print(f"Drone at {ip_address} is now active.")
            return True

        print(f"Attempting to connect to drone at {ip_address}...")
        connection = initialize_drone_connection(ip_address)
        if connection:
            self.drone_instances[ip_address] = connection
            print(f"Successfully connected to drone at {ip_address}.")
            if make_active or not self.active_drone_ip:
                self.active_drone_ip = ip_address
                print(f"Drone at {ip_address} is now active.")
            return True
        else:
            print(f"Failed to connect to drone at {ip_address}.")
            return False

    def set_active_drone(self, ip_address: str) -> bool:
        """Sets the active drone for commands."""
        if ip_address in self.drone_instances:
            self.active_drone_ip = ip_address
            print(f"Drone at {ip_address} is now the active drone.")
            return True
        else:
            print(f"Drone at {ip_address} is not connected. Cannot set as active.")
            return False

    def get_active_drone_instance(self) -> OpenDJI | None:
        """Returns the OpenDJI instance for the currently active drone."""
        if self.active_drone_ip and self.active_drone_ip in self.drone_instances:
            return self.drone_instances[self.active_drone_ip]
        # print("No active drone selected or the active IP is not in connected instances.")
        return None
        
    def list_connected_drones(self) -> list[str]:
        """Lists the IP addresses of all currently connected drones."""
        return list(self.drone_instances.keys())

    def disconnect_drone(self, ip_address: str, remove_active: bool = True):
        """Disconnects a specific drone."""
        if ip_address in self.drone_instances:
            print(f"Disconnecting drone at {ip_address}...")
            connection = self.drone_instances.pop(ip_address)
            close_drone_connection(connection) # Use the refactored close function
            print(f"Drone at {ip_address} disconnected.")
            if remove_active and self.active_drone_ip == ip_address:
                self.active_drone_ip = None
                print("Active drone was disconnected.")
                # Optionally, set another drone as active if available
                if self.drone_instances:
                    new_active_ip = list(self.drone_instances.keys())[0]
                    self.set_active_drone(new_active_ip)
        else:
            print(f"Drone at {ip_address} not found in connected instances.")

    def disconnect_all_drones(self):
        """Disconnects all connected drones."""
        print("Disconnecting all drones...")
        ips_to_disconnect = list(self.drone_instances.keys()) # Avoid modifying dict while iterating
        for ip_address in ips_to_disconnect:
            self.disconnect_drone(ip_address, remove_active=False) # remove_active=False to avoid issues if it's the current active one
        self.active_drone_ip = None
        print("All drones disconnected.")


    def run_query(self, query: str) -> str:
        """
        Runs a query using the initialized drone agent.
        Ensures an active drone is selected if the query implies drone action.
        """
        if not self.get_active_drone_instance():
             # Heuristic: if query mentions common drone actions, and no drone is active, prompt.
            action_keywords = ["take off", "land", "move", "rotate", "frame", "track", "fly", "drone"]
            if any(keyword in query.lower() for keyword in action_keywords):
                if self.drone_instances:
                    available_drones = self.list_connected_drones()
                    return (f"Error: The query seems to require drone action, but no drone is active. "
                            f"Please set an active drone. Connected drones: {available_drones}. "
                            f"Example: 'Connect to drone 192.168.1.120' or 'Set active drone 192.168.1.120'.")
                else:
                    return ("Error: The query seems to require drone action, but no drones are connected. "
                            "Please connect to a drone first. Example: 'Connect to drone 192.168.1.120'.")

        print(f"Sending query to agent for active drone {self.active_drone_ip or 'None'}: '{query}'")
        try:
            response = self.drone_agent.run(query)
            # print(f"Agent response:\n{response}") # Already printed by run_query in example
            return response
        except Exception as e:
            error_message = f"Error running agent query: {e}"
            print(error_message)
            # print(f"Please ensure the drone is connected and OpenDJI is set up correctly.")
            # print(f"Also, check your OPENROUTER_API_KEY (if required) and DRONE_IP_ADDR (current: {DRONE_IP_ADDR}) in the .env file.")
            return error_message

# --- Example Usage (Optional) ---
if __name__ == "__main__":
    print("Initializing Autonomous Drone Agent...")
    agent_instance = AutonomousDroneAgent()
    print("Autonomous Drone Agent Initialized.")

    # Example IPs - replace with your actual drone IPs
    drone1_ip = os.getenv("DRONE_IP_1", "192.168.1.115") 
    drone2_ip = os.getenv("DRONE_IP_2", "192.168.1.120") # Example for a second drone

    try:
        # Connect to the first drone
        if agent_instance.connect_to_drone(drone1_ip):
            print(f"Successfully connected to and activated drone at {drone1_ip}")

            # Example: Run a query for the first drone
            # print("\n--- Running Query on Drone 1 ---")
            # response = agent_instance.run_query(f"Take off the drone at {agent_instance.active_drone_ip}, fly forward a bit, then land.")
            # print(f"Response for {agent_instance.active_drone_ip}:\n{response}")

            # If you have a second drone and want to test switching:
            # if agent_instance.connect_to_drone(drone2_ip, make_active=True): # Connect and make active
            # print(f"Successfully connected to and activated drone at {drone2_ip}")
            # response_drone2 = agent_instance.run_query("What is the current frame from the drone?")
            # print(f"Response for {drone2_ip}:\n{response_drone2}")

            # Switch back to drone 1 if drone 2 was activated
            # if agent_instance.set_active_drone(drone1_ip):
            # response_drone1_again = agent_instance.run_query("Rotate the drone 90 degrees clockwise.")
            # print(f"Response for {drone1_ip} (again):\n{response_drone1_again}")
            
            # Example: YOLO Person Tracking on the active drone
            print("\n--- Running Example: YOLO Person Tracking (using wrapper) ---")
            # This uses the track_person_and_rotate_yolo_wrapper
            # The underlying yolo_tracker_old.track_person_and_rotate_yolo needs to be refactored
            # to accept drone_instance as the first argument.
            response = agent_instance.run_query("Use YOLO to track a person.") 
            print(f"Response for active drone ({agent_instance.active_drone_ip}):\n{response}")

        else:
            print(f"Failed to connect to drone at {drone1_ip}. Aborting example usage for this drone.")
            
        # Example: list connected drones
        print(f"Currently connected drones: {agent_instance.list_connected_drones()}")


    except Exception as e:
        print(f"An unexpected error occurred during example usage: {e}")
    finally:
        # The atexit handler in __init__ will call disconnect_all_drones()
        # You can also call it explicitly if needed earlier:
        # agent_instance.disconnect_all_drones() 
        print("Application finished. Drones should be disconnected by atexit handler.")
