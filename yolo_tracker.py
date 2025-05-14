import os
import time
import uuid
import json
from ultralytics import YOLO
from OpenDJI import OpenDJI # Ensure OpenDJI is installed and accessible
import numpy as np
from dotenv import load_dotenv
import atexit

# Load environment variables from .env file
load_dotenv()

# --- Configuration (specific to this module) ---
DRONE_IP_ADDR_YT = os.getenv("DRONE_IP_ADDR", "192.168.1.115")

# --- Global States for this module ---
yt_drone_connection = None
yt_yolo_model = None

# --- Logging Function (specific to this module) ---
def yt_log_message(log_file_name: str, message: str):
    """Logs a message to a file (in logs_yt directory) and prints it to the console."""
    logs_dir = "logs_yt" # Self-contained logging directory
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
            yt_log_message(f"yt_drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Attempting to connect to drone at IP: {DRONE_IP_ADDR_YT} (YT)...")
            yt_drone_connection = OpenDJI(DRONE_IP_ADDR_YT)
            yt_log_message(f"yt_drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Successfully connected to the drone (YT).")
        except Exception as e:
            yt_log_message(f"yt_drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Failed to connect to the drone (YT): {e}")
            yt_drone_connection = None
    return yt_drone_connection

def yt_close_drone_connection():
    """Closes this module's global drone connection if it's open."""
    global yt_drone_connection
    if yt_drone_connection:
        try:
            yt_log_message(f"yt_drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Closing drone connection (YT)...")
            yt_drone_connection.close()
            yt_log_message(f"yt_drone_connection_log_{uuid.uuid4().hex[:8]}.txt", "Drone connection closed (YT).")
        except Exception as e:
            yt_log_message(f"yt_drone_connection_log_{uuid.uuid4().hex[:8]}.txt", f"Error closing drone connection (YT): {e}")
        finally:
            yt_drone_connection = None

# Register the close function for this module to be called on exit
atexit.register(yt_close_drone_connection)

# --- YOLO Model Initialization (specific to this module) ---
def yt_initialize_yolo_model():
    """Initializes this module's global YOLO model."""
    global yt_yolo_model
    if yt_yolo_model is None:
        yt_log_message(f"yt_yolo_model_log_{uuid.uuid4().hex[:8]}.txt", "Starting YOLO model initialization (YT)...")
        try:
            yt_log_message(f"yt_yolo_model_log_{uuid.uuid4().hex[:8]}.txt", "Attempting to load YOLO model (yolov8n.pt) (YT)...")
            start_time = time.time()
            yt_yolo_model = YOLO("yolov8n.pt") # Ensure 'yolov8n.pt' is accessible
            load_time = time.time() - start_time
            yt_log_message(f"yt_yolo_model_log_{uuid.uuid4().hex[:8]}.txt", f"YOLO model loaded successfully in {load_time:.2f} seconds (YT).")
        except Exception as e:
            yt_log_message(f"yt_yolo_model_log_{uuid.uuid4().hex[:8]}.txt", f"Error loading YOLO model (YT): {e}")
            yt_log_message(f"yt_yolo_model_log_{uuid.uuid4().hex[:8]}.txt", "Please ensure the YOLO model file (e.g., 'yolov8n.pt') is available (YT).")
            yt_yolo_model = None
    return yt_yolo_model

# --- YOLO Analysis Function (specific to this module) ---
def yt_analyze_image_with_yolo(image_frame, log_file_name: str):
    """Analyzes an image frame using this module's YOLO and returns results."""
    global yt_yolo_model
    if yt_yolo_model is None:
        yt_initialize_yolo_model()
        if yt_yolo_model is None:
            yt_log_message(log_file_name, "Error: YOLO model not initialized (YT) within yt_analyze_image_with_yolo.")
            return None, "Error: YOLO model not initialized (YT)."

    yt_log_message(log_file_name, "Starting YOLO object detection (YT)")
    try:
        height, width, channels = image_frame.shape
        yt_log_message(log_file_name, f"Input image for YOLO (YT): {width}x{height}x{channels}")
        start_time = time.time()
        results = yt_yolo_model(image_frame)
        detection_time = time.time() - start_time
        yt_log_message(log_file_name, f"YOLO detection completed in {detection_time:.2f} seconds (YT)")

        if results and results[0].boxes:
            total_objects = len(results[0].boxes)
            yt_log_message(log_file_name, f"YOLO detected {total_objects} objects (YT)")
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
                yt_log_message(log_file_name, summary)
                detection_summary.append(summary)
            return results, "\\n".join(detection_summary)
        else:
            yt_log_message(log_file_name, "No objects detected by YOLO (YT)")
            return None, "No objects detected by YOLO (YT)."
    except Exception as e:
        yt_log_message(log_file_name, f"Error during YOLO analysis (YT): {e}")
        return None, f"Error during YOLO analysis (YT): {e}"

def track_person_and_rotate_yolo(max_iterations: int = 300000000000, seconds_per_iteration: float = 0.2) -> str:
    """
    Commands the drone to take off, then continuously uses YOLO object detection
    to find a person, calculate their position relative to the frame center,
    and issue rotation (rcw) and forward (bf) commands to track them in real-time.
    Uses self-contained utilities (YT suffixed) for logging, drone, and YOLO management.
    Finally, commands the drone to land. Aims for lower latency than LLM-based tracking.

    Args:
        max_iterations: The maximum number of tracking iterations.
        seconds_per_iteration: The target total cycle time for each iteration.

    Returns:
        str: A message indicating the result of the tracking sequence.
    """
    log_file_name = f"yt_tracking_log_{uuid.uuid4().hex[:8]}.txt" # Specific log for this run
    yt_log_message(log_file_name, "Initiating YOLO-based automated person tracking sequence (YT)...")
    global yt_drone_connection, yt_yolo_model # Referencing this module's globals

    if yt_drone_connection is None:
        yt_initialize_drone_connection()
        if yt_drone_connection is None:
            yt_log_message(log_file_name, "Error: Drone connection not established (YT). Cannot start tracking.")
            return "Error: Drone connection not established (YT). Cannot start tracking."

    if yt_yolo_model is None:
        yt_initialize_yolo_model()
        if yt_yolo_model is None:
            yt_log_message(log_file_name, "Error: YOLO model not initialized (YT). Cannot start tracking.")
            return "Error: YOLO model not initialized (YT). Cannot start tracking."

    try:
        drone = yt_drone_connection # Use this module's connection

        result = drone.enableControl(True)
        yt_log_message(log_file_name, f"Enable SDK control command sent (YT). Drone response: {result}")
        
        yt_log_message(log_file_name, "Sending takeoff command (YT)...")
        takeoff_result = drone.takeoff(True)
        yt_log_message(log_file_name, f"Takeoff result (YT): {takeoff_result}")
        if "error" in str(takeoff_result).lower() or "failed" in str(takeoff_result).lower():
            yt_log_message(log_file_name, f"Takeoff failed, cannot start tracking (YT): {takeoff_result}")
            return f"Takeoff failed, cannot start tracking (YT): {takeoff_result}"

        yt_log_message(log_file_name, "Stabilizing after takeoff (YT)...")
        time.sleep(10)

        yt_log_message(log_file_name, f"Starting YOLO person tracking for up to {max_iterations} iterations (YT).")
        
        MAX_SPEED = 1.0 
        CENTER_THRESHOLD_PERCENT = 0.1 
        DEGREES_PER_SECOND_ROTATION = 180.0 / 3.0
        METERS_PER_SECOND_FORWARD = 1.0 / 3.0
        FOV_HORIZONTAL_DEGREES = 60.0
        DESIRED_FORWARD_DISTANCE_M = 1.0
        SCAN_ANGLE_DEGREES = 15.0
        MIN_ACTION_DURATION = 0.05
        MAX_ROTATION_ACTION_DURATION = 3.0
        MAX_FORWARD_ACTION_DURATION = 6.0

        try:
            for i in range(max_iterations):
                iteration_start_time = time.time()
                yt_log_message(log_file_name, f"Tracking iteration {i+1}/{max_iterations} (YT)...")
                person_found_this_iteration = False
                
                try:
                    frame_np = drone.getFrame()
                    if frame_np is None:
                        yt_log_message(log_file_name, "No frame available (YT). Skipping iteration.")
                        time.sleep(seconds_per_iteration) 
                        continue

                    H, W, _ = frame_np.shape
                    center_x = W / 2.0

                    yolo_results_obj, yolo_summary = yt_analyze_image_with_yolo(frame_np, log_file_name) 

                    if yolo_results_obj and yolo_results_obj[0].boxes:
                        best_person_box = None
                        max_conf = 0.0
                        for box in yolo_results_obj[0].boxes:
                            class_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = yt_yolo_model.names[class_id]
                            if class_name == 'person' and conf > max_conf:
                                max_conf = conf
                                best_person_box = box.xyxy[0].cpu().numpy()
                                person_found_this_iteration = True
                        
                        if person_found_this_iteration:
                            x1, y1, x2, y2 = best_person_box
                            px1 = float(x1)
                            px2 = float(x2)
                            person_cx = (px1 + px2) / 2.0
                            dx = person_cx - center_x
                            current_center_threshold_pixels = W * CENTER_THRESHOLD_PERCENT

                            if abs(dx) > current_center_threshold_pixels:
                                angle_to_correct_degrees = (dx / W) * FOV_HORIZONTAL_DEGREES
                                current_rcw_command = MAX_SPEED if dx > 0 else -MAX_SPEED
                                rotation_duration_calculated = abs(angle_to_correct_degrees) / DEGREES_PER_SECOND_ROTATION
                                rotation_duration_actual = max(MIN_ACTION_DURATION, min(rotation_duration_calculated, MAX_ROTATION_ACTION_DURATION))
                                yt_log_message(log_file_name, f"  Person off-center (dx={dx:.1f}px, angle={angle_to_correct_degrees:.1f}deg). Rotating: rcw={current_rcw_command:.2f} for {rotation_duration_actual:.2f}s (YT)")
                                drone.move(current_rcw_command, 0, 0, 0)
                                time.sleep(rotation_duration_actual)
                                drone.move(0, 0, 0, 0)
                            else:
                                forward_duration_calculated = DESIRED_FORWARD_DISTANCE_M / METERS_PER_SECOND_FORWARD
                                forward_duration_actual = max(MIN_ACTION_DURATION, min(forward_duration_calculated, MAX_FORWARD_ACTION_DURATION))
                                current_bf_command = MAX_SPEED
                                yt_log_message(log_file_name, f"  Person centered. Moving forward: bf={current_bf_command:.2f} for {forward_duration_actual:.2f}s (dist: {DESIRED_FORWARD_DISTANCE_M:.2f}m) (YT)")
                                drone.move(0, 0, 0, current_bf_command)
                                time.sleep(forward_duration_actual)
                                drone.move(0, 0, 0, 0)
                    
                    if not person_found_this_iteration:
                        scan_rotation_duration = SCAN_ANGLE_DEGREES / DEGREES_PER_SECOND_ROTATION
                        scan_rotation_actual = max(MIN_ACTION_DURATION, min(scan_rotation_duration, MAX_ROTATION_ACTION_DURATION))
                        current_rcw_scan_command = MAX_SPEED
                        yt_log_message(log_file_name, f"  No person detected. Scanning clockwise ({SCAN_ANGLE_DEGREES}deg): rcw={current_rcw_scan_command:.2f} for {scan_rotation_actual:.2f}s (YT)")
                        drone.move(current_rcw_scan_command, 0, 0, 0)
                        time.sleep(scan_rotation_actual)
                        drone.move(0, 0, 0, 0)

                except Exception as e:
                    yt_log_message(log_file_name, f"Error in YOLO tracking iteration {i+1} (YT): {str(e)}")
                    try:
                        drone.move(0, 0, 0, 0)
                    except Exception as stop_e:
                        yt_log_message(log_file_name, f"Error stopping drone after iteration error (YT): {stop_e}")
            
            yt_log_message(log_file_name, "Max iterations reached or tracking stopped (YT).")
            yt_log_message(log_file_name, "Landing the drone (YT)...")
            drone.move(0, 0, 0, 0) 
            land_result = drone.land(True)
            yt_log_message(log_file_name, f"Landing result (YT): {land_result}")
            return f"YOLO person tracking completed after {i + 1} iterations (YT)."

        except KeyboardInterrupt:
            yt_log_message(log_file_name, "User interrupted (Ctrl+C) (YT). Landing the drone...")
            try:
                drone.move(0, 0, 0, 0)
                land_result = drone.land(True)
                yt_log_message(log_file_name, f"Landing result (YT): {land_result}")
            except Exception as e_land:
                yt_log_message(log_file_name, f"Error during landing after interruption (YT): {e_land}")
            return "YOLO person tracking interrupted by user (YT). Drone landed."

    except Exception as e:
        yt_log_message(log_file_name, f"An overall error occurred during the YOLO track_person sequence (YT): {e}")
        try:
            yt_log_message(log_file_name, "Attempting emergency land (YT)...")
            if yt_drone_connection:
                yt_drone_connection.move(0, 0, 0, 0) 
                yt_drone_connection.land(True)
        except Exception as land_e:
            yt_log_message(log_file_name, f"Failed to execute emergency land (YT): {land_e}")
        return f"Error during YOLO tracking sequence (YT): {e}" 