import os
import time
import uuid
import json
from ultralytics import YOLO
from OpenDJI import OpenDJI # Ensure OpenDJI is installed and accessible
import numpy as np
import cv2
from dotenv import load_dotenv
import atexit

# Load environment variables from .env file
load_dotenv()

# --- Configuration (specific to this module) ---
DRONE_IP_ADDR_YT = os.getenv("DRONE_IP_ADDR", "192.168.1.115")
# DRONE_IP_ADDR_YT = os.getenv("DRONE_IP_ADDR", "0.0.0.0")

# --- Global States for this module ---
yt_drone_connection = None
yt_yolo_model = None

# --- Logging Function (specific to this module) ---
def log_message(log_file_name: str, message: str):
    """Logs a message to a file (in logs_yt directory) and prints it to the console."""
    logs_dir = "logs" # Self-contained logging directory
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    file_path = os.path.join(logs_dir, log_file_name)
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    formatted_message = f"[{timestamp}] {message}"
    
    with open(file_path, "a") as f:
        f.write(formatted_message + "\\n")
    print(formatted_message)

# --- Drone Connection Management (specific to this module) ---
def yt_initialize_drone_connection():
    """Initializes this module's global drone connection."""
    global yt_drone_connection
    if yt_drone_connection is None:
        try:
            log_message(f"yt_drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Attempting to connect to drone at IP: {DRONE_IP_ADDR_YT} (YT)...")
            yt_drone_connection = OpenDJI(DRONE_IP_ADDR_YT)
            log_message(f"yt_drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Successfully connected to the drone (YT).")
        except Exception as e:
            log_message(f"yt_drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Failed to connect to the drone (YT): {e}")
            yt_drone_connection = None
    return yt_drone_connection

def yt_close_drone_connection():
    """Closes this module's global drone connection if it's open."""
    global yt_drone_connection
    if yt_drone_connection:
        try:
            log_message(f"yt_drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Closing drone connection (YT)...")
            yt_drone_connection.close()
            log_message(f"yt_drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Drone connection closed (YT).")
        except Exception as e:
            log_message(f"yt_drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Error closing drone connection (YT): {e}")
        finally:
            yt_drone_connection = None

# Register the close function for this module to be called on exit
atexit.register(yt_close_drone_connection)

# --- YOLO Model Initialization (specific to this module) ---
def yt_initialize_yolo_model():
    """Initializes this module's global YOLO model."""
    global yt_yolo_model
    if yt_yolo_model is None:
        log_message(f"yt_yolo_model_log_{uuid.uuid4().hex[:8]}.txt", "Starting YOLO model initialization (YT)...")
        try:
            log_message(f"yt_yolo_model_log_{uuid.uuid4().hex[:8]}.txt", "Attempting to load YOLO model (yolov8n.pt) (YT)...")
            start_time = time.time()
            yt_yolo_model = YOLO("yolov8n.pt") # Ensure 'yolov8n.pt' is accessible
            load_time = time.time() - start_time
            log_message(f"yt_yolo_model_log_{uuid.uuid4().hex[:8]}.txt", f"YOLO model loaded successfully in {load_time:.2f} seconds (YT).")
        except Exception as e:
            log_message(f"yt_yolo_model_log_{uuid.uuid4().hex[:8]}.txt", f"Error loading YOLO model (YT): {e}")
            log_message(f"yt_yolo_model_log_{uuid.uuid4().hex[:8]}.txt", "Please ensure the YOLO model file (e.g., 'yolov8n.pt') is available (YT).")
            yt_yolo_model = None
    return yt_yolo_model

# --- YOLO Analysis Function (specific to this module) ---
def yt_analyze_image_with_yolo(image_frame, log_file_name: str):
    """Analyzes an image frame using this module's YOLO and returns results."""
    global yt_yolo_model
    if yt_yolo_model is None:
        yt_initialize_yolo_model()
        if yt_yolo_model is None:
            log_message(log_file_name, "Error: YOLO model not initialized (YT) within yt_analyze_image_with_yolo.")
            return None, "Error: YOLO model not initialized (YT)."

    log_message(log_file_name, "Starting YOLO object detection (YT)")
    try:
        height, width, channels = image_frame.shape
        log_message(log_file_name, f"Input image for YOLO (YT): {width}x{height}x{channels}")
        start_time = time.time()
        results = yt_yolo_model(image_frame)
        detection_time = time.time() - start_time
        log_message(log_file_name, f"YOLO detection completed in {detection_time:.2f} seconds (YT)")

        if results and results[0].boxes:
            total_objects = len(results[0].boxes)
            log_message(log_file_name, f"YOLO detected {total_objects} objects (YT)")
            detected_classes = {}
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = yt_yolo_model.names[class_id]
                confidence = float(box.conf[0])
                if class_name in detected_classes:
                    detected_classes[class_name].append(confidence)
                else:
                    detected_classes[class_name] = [confidence]
            detection_summary = []
            for class_name, confidences in detected_classes.items():
                avg_conf = sum(confidences) / len(confidences)
                summary = f"Detected {len(confidences)}x {class_name} (avg conf: {avg_conf:.2f}) (YT)"
                log_message(log_file_name, summary)
                detection_summary.append(summary)
            return results, "\\n".join(detection_summary)
        else:
            log_message(log_file_name, "No objects detected by YOLO (YT)")
            return None, "No objects detected by YOLO (YT)."
    except Exception as e:
        log_message(log_file_name, f"Error during YOLO analysis (YT): {e}")
        return None, f"Error during YOLO analysis (YT): {e}"

def track_person_realtime_yolo(max_iterations: int = 300000000000, seconds_per_iteration: float = 0.1) -> str:
    """
    Commands the drone to take off, then continuously uses YOLO object detection
    to find a person, calculate their position relative to the frame center,
    and issue rotation (rcw) and forward (bf) commands to track them in real-time
    by repeatedly setting drone.move() without intermediate stops.
    Uses self-contained utilities (YT suffixed) for logging, drone, and YOLO management.
    Finally, commands the drone to land. Aims for lower latency and smoother tracking.
    Saves image frames and logs detailed command info.

    Args:
        max_iterations: The maximum number of tracking iterations.
        seconds_per_iteration: The target total cycle time for each iteration.

    Returns:
        str: A message indicating the result of the tracking sequence.
    """
    log_file_name = f"yt_realtime_tracking_log_{uuid.uuid4().hex[:8]}.txt" # Specific log for this run

    # Create a directory for saving frames for this run
    base_log_name = os.path.splitext(os.path.basename(log_file_name))[0]
    frames_dir = os.path.join(os.path.dirname(log_file_name) or "logs", base_log_name + "_frames")
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    log_message(log_file_name, f"Frames for this run will be saved in: {frames_dir} (YT_RT)")
    
    log_message(log_file_name, "Initiating YOLO-based REAL-TIME automated person tracking sequence (YT_RT)...")
    global yt_drone_connection, yt_yolo_model # Referencing this module's globals

    if yt_drone_connection is None:
        yt_initialize_drone_connection()
        if yt_drone_connection is None:
            log_message(log_file_name, "Error: Drone connection not established (YT_RT). Cannot start tracking.")
            return "Error: Drone connection not established (YT_RT). Cannot start tracking."

    if yt_yolo_model is None:
        yt_initialize_yolo_model()
        if yt_yolo_model is None:
            log_message(log_file_name, "Error: YOLO model not initialized (YT_RT). Cannot start tracking.")
            return "Error: YOLO model not initialized (YT_RT). Cannot start tracking."

    drone = yt_drone_connection # Use this module's connection

    result = drone.enableControl(True)
    log_message(log_file_name, f"Enable SDK control command sent (YT_RT). Drone response: {result}")
    
    log_message(log_file_name, "Sending takeoff command (YT_RT)...")
    takeoff_result = drone.takeoff(True)
    log_message(log_file_name, f"Takeoff result (YT_RT): {takeoff_result}")
    if "error" in str(takeoff_result).lower() or "failed" in str(takeoff_result).lower():
        log_message(log_file_name, f"Takeoff failed, cannot start tracking (YT_RT): {takeoff_result}")
        return f"Takeoff failed, cannot start tracking (YT_RT): {takeoff_result}"

    log_message(log_file_name, "Stabilizing after takeoff (YT_RT)...")
    time.sleep(9.5) # Keep stabilization period

    log_message(log_file_name, f"Starting YOLO real-time person tracking for up to {max_iterations} iterations (YT_RT).")
    
    drone.move(rcw=0.0, du=0.0, lr=0.0, bf=0.2)
    time.sleep(20)
    drone.move(rcw=0.0, du=0.0, lr=0.0, bf=0.5)
    time.sleep(20)
    drone.move(rcw=0.0, du=0.0, lr=0.0, bf=1.0)
    time.sleep(20)
    drone.move(rcw=1.0, du=0.0, lr=0.0, bf=0.0)
    time.sleep(20)
    drone.move(rcw=0.2, du=0.0, lr=0.0, bf=0.0)
    time.sleep(20)
    drone.move(rcw=0.5, du=0.0, lr=0.0, bf=0.0)
    time.sleep(20)
    drone.land(True)

if __name__ == "__main__":
    track_person_realtime_yolo()