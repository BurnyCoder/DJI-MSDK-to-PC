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

def track_person_yolo(max_iterations: int = 300000000000, seconds_per_iteration: float = 0.1) -> str:
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

    try:
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
        
        MAX_ROTATE_SPEED = 0.15  # Renamed from MAX_SPEED, value from FPVdemo.py ROTATE_VALUE
        MAX_FORWARD_SPEED = 0.015 # Value from FPVdemo.py MOVE_VALUE
        CENTER_THRESHOLD_PERCENT = 0.1 
        FOV_HORIZONTAL_DEGREES = 60.0
        # DESIRED_FORWARD_DISTANCE_M = 1.0 # Less directly applicable for continuous control logic but useful for context
        # SCAN_ANGLE_DEGREES = 15.0 # Scanning is now continuous rotation at MAX_SPEED

        try:
            for i in range(max_iterations):
                iteration_start_time = time.time()
                log_message(log_file_name, f"Tracking iteration {i+1}/{max_iterations} (YT_RT)...")
                
                current_rcw_command = 0.0
                current_bf_command = 0.0
                person_found_this_iteration = False
                frame_saved_path = None
                dx_for_filename = "NA" # Initialize for filename
                log_decision_message_template = None # For deferred logging
                log_decision_values = {} # For deferred logging
                
                try:
                    frame_np = drone.getFrame()
                    if frame_np is None:
                        log_message(log_file_name, f"Decision: No frame available. Action: Holding position. Command: rcw=0, lr=0, ud=0, bf=0. Frame: {frame_saved_path} (YT_RT)")
                        drone.move(0, 0, 0, 0) # ADDED BACK: Explicitly hold if no frame
                        # Calculate sleep time to maintain iteration cadence
                        elapsed_processing_time = time.time() - iteration_start_time
                        sleep_duration = seconds_per_iteration - elapsed_processing_time
                        if sleep_duration > 0:
                            time.sleep(sleep_duration)
                        continue
                    else:
                        # Save frame
                        current_frame_filename = f"iter_{i+1}_frame.jpg"
                        frame_saved_path = os.path.join(frames_dir, current_frame_filename)
                        # cv2.imwrite(frame_saved_path, frame_np)
                        # log_message(log_file_name, f"  Frame saved: {frame_saved_path} (YT_RT)") # Optional

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
                            dx_for_filename = f"{dx:.1f}".replace('.', 'p').replace('-', 'neg') # Update dx_for_filename
                            current_center_threshold_pixels = W * CENTER_THRESHOLD_PERCENT

                            if abs(dx) > current_center_threshold_pixels:
                                angle_to_correct_degrees = (dx / W) * FOV_HORIZONTAL_DEGREES # Informational
                                current_rcw_command = MAX_ROTATE_SPEED if dx > 0 else -MAX_ROTATE_SPEED
                                # current_bf_command remains 0.0
                                log_decision_message_template = "  Decision: Person off-center (dx={{dx:.1f}}px, angle={{angle:.1f}}deg). Action: Adjusting rotation. Current command values: rcw={{rcw:.2f}}, bf={{bf:.2f}}. Frame: {{frame_path}} (YT_RT)"
                                log_decision_values = {'dx': dx, 'angle': angle_to_correct_degrees, 'rcw': current_rcw_command, 'bf': current_bf_command}
                            else:
                                # current_rcw_command remains 0.0
                                current_bf_command = MAX_FORWARD_SPEED
                                log_decision_message_template = "  Decision: Person centered. Action: Adjusting forward. Current command values: rcw={{rcw:.2f}}, bf={{bf:.2f}}. Frame: {{frame_path}} (YT_RT)"
                                log_decision_values = {'rcw': current_rcw_command, 'bf': current_bf_command}
                    
                    if not person_found_this_iteration:
                        current_rcw_command = MAX_ROTATE_SPEED # Scan by rotating
                        # current_bf_command remains 0.0
                        log_decision_message_template = "  Decision: No person detected. Action: Scanning. Current command values: rcw={{rcw:.2f}}, bf={{bf:.2f}}. Frame: {{frame_path}} (YT_RT)"
                        log_decision_values = {'rcw': current_rcw_command, 'bf': current_bf_command}

                    # Save frame with detailed name including command values
                    rcw_str = f"{current_rcw_command:.2f}".replace('.', 'p').replace('-', 'neg')
                    bf_str = f"{current_bf_command:.2f}".replace('.', 'p').replace('-', 'neg')
                    person_str = "T" if person_found_this_iteration else "F"
                    # dx_for_filename is already formatted and safe

                    detailed_frame_filename = f"rcw{rcw_str}_bf{bf_str}_person{person_str}_dx{dx_for_filename}_iter{i+1}.jpg"
                    frame_saved_path = os.path.join(frames_dir, detailed_frame_filename)
                    cv2.imwrite(frame_saved_path, frame_np)
                    
                    # Log the decision using the template and the new frame_saved_path
                    if log_decision_message_template:
                        log_decision_values['frame_path'] = frame_saved_path
                        log_message(log_file_name, log_decision_message_template.format(**log_decision_values))

                    log_message(log_file_name, f"Executing drone.move(rcw={current_rcw_command:.2f}, lr=0, ud=0, bf={current_bf_command:.2f}) (YT_RT)")
                    drone.move(current_rcw_command, 0, 0, current_bf_command)

                except Exception as e_iter:
                    log_message(log_file_name, f"Error in YOLO real-time tracking iteration {i+1} (YT_RT): {str(e_iter)}. Frame: {frame_saved_path}")
                    try: # ADDED BACK: Attempt to stop movement on iteration error
                        drone.move(0, 0, 0, 0) 
                    except Exception as stop_e:
                        log_message(log_file_name, f"Error stopping drone after iteration error (YT_RT): {stop_e}")
            
            log_message(log_file_name, "Max iterations reached or tracking stopped (YT_RT).")
            log_message(log_file_name, "Stopping drone and landing (YT_RT)...")
            drone.move(0, 0, 0, 0) 
            land_result = drone.land(True)
            log_message(log_file_name, f"Landing result (YT_RT): {land_result}")
            return f"YOLO real-time person tracking completed after {i + 1} iterations (YT_RT)."

        except KeyboardInterrupt:
            log_message(log_file_name, "User interrupted (Ctrl+C) (YT_RT). Stopping drone and landing...")
            try:
                drone.move(0, 0, 0, 0)
                land_result = drone.land(True)
                log_message(log_file_name, f"Landing result (YT_RT): {land_result}")
            except Exception as e_land:
                log_message(log_file_name, f"Error during landing after interruption (YT_RT): {e_land}")
            return "YOLO real-time person tracking interrupted by user (YT_RT). Drone landed."

    except Exception as e:
        log_message(log_file_name, f"An overall error occurred during the YOLO real-time track_person sequence (YT_RT): {e}")
        try:
            log_message(log_file_name, "Attempting emergency stop and land (YT_RT)...")
            if yt_drone_connection:
                yt_drone_connection.move(0, 0, 0, 0) 
                yt_drone_connection.land(True)
        except Exception as land_e:
            log_message(log_file_name, f"Failed to execute emergency land (YT_RT): {land_e}")
        return f"Error during YOLO real-time tracking sequence (YT_RT): {e}" 

if __name__ == "__main__":
    track_person_yolo()