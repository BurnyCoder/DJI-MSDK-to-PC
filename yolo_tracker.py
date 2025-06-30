"""
This application demonstrates real-time person tracking using a DJI drone and YOLO object detection.
It uses a multi-threaded approach to:
1. Process video frames from the drone and detect people using YOLO
2. Calculate movement commands to keep detected people centered in the frame
3. Send continuous movement commands to the drone for smooth tracking, such as rotating and moving forward

The system runs two parallel threads:
- A processing thread that handles frame capture, YOLO detection, and movement calculation
- A movement thread that continuously sends commands to the drone

Key features:
- Automatic takeoff and landing
- Real-time person detection and tracking
- Frame-by-frame logging with detection details
- Graceful shutdown with keyboard interrupt ('x' key)
- Configurable movement parameters via environment variables
"""

from OpenDJI import OpenDJI
from ultralytics import YOLO
import time
import os

import keyboard
import cv2
import numpy as np
import threading

# IP address of the connected android device
# IP_ADDR = os.environ.get("IP_ADDR", "192.168.1.115")
# IP_ADDR = os.environ.get("IP_ADDR", "100.93.47.145")
# IP_ADDR = os.environ.get("IP_ADDR", "172.18.112.1")
IP_ADDR = os.environ.get("IP_ADDR", "100.85.47.22")

# Movement factors
MOVE_VALUE = float(os.environ.get("MOVE_VALUE", "0.1"))
ROTATE_VALUE = float(os.environ.get("ROTATE_VALUE", "0.2"))
CENTER_THRESHOLD_PERCENT = float(os.environ.get("CENTER_THRESHOLD_PERCENT", "0.2"))

# Thread timing constants
PROCESSING_THREAD_INTERVAL = float(os.environ.get("PROCESSING_THREAD_INTERVAL", "0.1"))  # Interval for frame processing and YOLO thread
MOVEMENT_COMMAND_INTERVAL = float(os.environ.get("MOVEMENT_COMMAND_INTERVAL", "0.02"))  # Interval for sending movement commands

# Create blank frame
BLANK_FRAME = np.zeros((1080, 1920, 3))
BLNAK_FRAME = cv2.putText(BLANK_FRAME, "No Image", (200, 300),
                          cv2.FONT_HERSHEY_DUPLEX, 10,
                          (255, 255, 255), 15)

# Create frames directory for logging
FRAMES_DIR = "yolo_frames"
if not os.path.exists(FRAMES_DIR):
    os.makedirs(FRAMES_DIR)
print(f"Frames will be saved in directory: {FRAMES_DIR}")

# Initialize YOLO model
try:
    yolo_model = YOLO("yolov8n.pt")
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure 'yolov8n.pt' is accessible and ultralytics is installed.")
    yolo_model = None

# --- Shared data between threads ---
shared_data_lock = threading.Lock()
latest_movement_commands = {
    "pitch": 0.0,
    "yaw": 0.0,
    "roll": 0.0,
    "ascent": 0.0,
    "person_found": False # Added for logging in movement thread
}
stop_threads_event = threading.Event()
# --- End of Shared data ---

# --- Thread 1: Frame Processing and YOLO ---
def frame_processing_and_yolo_thread_func(drone_obj, yolo_model_obj, blank_frame_img, frames_dir_path):
    print("Processing thread started.")
    iter_count = 0 # Initialize iteration counter for this thread

    while not stop_threads_event.is_set():
        start_processing_time = time.time()
        iter_count += 1

        current_frame_data = drone_obj.getFrame()
        frame_for_processing = blank_frame_img.copy() if current_frame_data is None else current_frame_data.copy()

        # Local movement variables for this cycle's calculation
        calculated_pitch = 0.0
        calculated_yaw = 0.0
        calculated_roll = 0.0
        calculated_ascent = 0.0 # Assuming ascent and roll are not dynamically changed by YOLO yet
        dx_val_for_filename = None
        person_detected_this_frame = False

        if not np.array_equal(frame_for_processing, blank_frame_img) and frame_for_processing.shape[0] > 0 and frame_for_processing.shape[1] > 0:
            H, W, _ = frame_for_processing.shape
            frame_center_x = W / 2.0
            current_center_threshold_pixels = W * CENTER_THRESHOLD_PERCENT

            yolo_results = yolo_model_obj(frame_for_processing, verbose=False)
            
            if yolo_results and yolo_results[0].boxes:
                for box in yolo_results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = yolo_model_obj.names[class_id]
                    
                    if class_name == 'person':
                        person_coords = box.xyxy[0].cpu().numpy() 
                        px1, _, px2, _ = person_coords
                        person_cx = (px1 + px2) / 2.0
                        dx_val = person_cx - frame_center_x
                        dx_val_for_filename = dx_val # Save for filename

                        if abs(dx_val) > current_center_threshold_pixels:
                            calculated_yaw = ROTATE_VALUE if dx_val > 0 else -ROTATE_VALUE
                            calculated_pitch = 0.0
                        else:
                            calculated_yaw = 0.0
                            calculated_pitch = MOVE_VALUE
                        
                        person_detected_this_frame = True
                        break 

            if not person_detected_this_frame:
                calculated_yaw = ROTATE_VALUE # Scan
                calculated_pitch = 0.0
        else: # Blank frame or invalid shape
            calculated_yaw = 0.0
            calculated_pitch = 0.0
            # person_detected_this_frame remains False
        
        # Update shared movement commands
        with shared_data_lock:
            latest_movement_commands["pitch"] = calculated_pitch
            latest_movement_commands["yaw"] = calculated_yaw
            latest_movement_commands["roll"] = calculated_roll
            latest_movement_commands["ascent"] = calculated_ascent
            latest_movement_commands["person_found"] = person_detected_this_frame

        # Image Saving Logic (using calculated values before lock or from local vars)
        if not np.array_equal(frame_for_processing, blank_frame_img) and frame_for_processing.shape[0] > 0 and frame_for_processing.shape[1] > 0:
            yaw_str = f"{calculated_yaw:.2f}".replace('.', 'p').replace('-', 'neg')
            pitch_str = f"{calculated_pitch:.2f}".replace('.', 'p').replace('-', 'neg')
            person_str = "T" if person_detected_this_frame else "F"
            
            dx_filename_part = "NA"
            if person_detected_this_frame and dx_val_for_filename is not None:
                dx_filename_part = f"{dx_val_for_filename:.1f}".replace('.', 'p').replace('-', 'neg')

            detailed_frame_filename = f"yaw{yaw_str}_pitch{pitch_str}_person{person_str}_dx{dx_filename_part}_iter{iter_count}.jpg"
            frame_saved_path = os.path.join(frames_dir_path, detailed_frame_filename)
            
            try:
                cv2.imwrite(frame_saved_path, frame_for_processing)
            except Exception as e_save:
                print(f"Error saving frame {frame_saved_path}: {e_save}")

        # Ensure this thread runs approximately every 0.5 seconds
        elapsed_time = time.time() - start_processing_time
        sleep_time = PROCESSING_THREAD_INTERVAL - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
    print("Processing thread stopped.")

# --- Thread 2: Drone Movement ---
def drone_movement_thread_func(drone_obj):
    print("Movement thread started.")
    while not stop_threads_event.is_set():
        local_pitch, local_yaw, local_roll, local_ascent, local_person_found = 0.0, 0.0, 0.0, 0.0, False
        with shared_data_lock:
            local_pitch = latest_movement_commands["pitch"]
            local_yaw = latest_movement_commands["yaw"]
            local_roll = latest_movement_commands["roll"]
            local_ascent = latest_movement_commands["ascent"]
            local_person_found = latest_movement_commands["person_found"] # Get for logging

        # Print current movement values being sent to the drone
        print(f"Movement: pitch={local_pitch:.2f}, yaw={local_yaw:.2f}, roll={local_roll:.2f}, ascent={local_ascent:.2f}, person_found={local_person_found}")
        drone_obj.move(local_pitch, local_yaw, local_roll, local_ascent)
        time.sleep(MOVEMENT_COMMAND_INTERVAL) # Send command every 0.02 seconds
    print("Movement thread stopped.")

# Connect to the drone
with OpenDJI(IP_ADDR) as drone:
    if yolo_model is None:
        print("YOLO model not loaded. Exiting application.")
    else:
        result = drone.enableControl(True)
        print(f"Enable SDK control command sent. Drone response: {result}")

        print("Taking off...")
        drone.takeoff(True)
        time.sleep(10)
        print("Taking off complete")

        # Create and start threads
        processing_thread = threading.Thread(target=frame_processing_and_yolo_thread_func, 
                                             args=(drone, yolo_model, BLANK_FRAME, FRAMES_DIR))
        movement_thread = threading.Thread(target=drone_movement_thread_func, args=(drone,))

        processing_thread.start()
        movement_thread.start()

        print("Press 'x' to stop and land the drone.")
        while not stop_threads_event.is_set():
            if keyboard.is_pressed('x'):
                print("'x' pressed, signaling threads to stop...")
                stop_threads_event.set()
                break
            # Check if threads are alive, if not, signal stop (safety)
            if not processing_thread.is_alive() or not movement_thread.is_alive():
                print("A thread has unexpectedly stopped. Signaling all threads to stop.")
                stop_threads_event.set()
                break
            time.sleep(0.1) # Main thread polling interval

        print("Waiting for threads to complete...")
        processing_thread.join()
        movement_thread.join()
        print("Threads have completed.")

    print("Landing drone...")
    drone.land(True)
    print("Drone landed. Exiting program.")

# Original main loop is now replaced by the threaded structure above.
# The rest of the original code related to the single loop is removed or integrated into threads.

# Comment out or remove the old loop structure:
# iter_count = 0 # Initialize iteration counter
# Press 'x' to close the program
# print("Press 'x' to close the program")
# while not keyboard.is_pressed('x') and yolo_model is not None:
#     iter_count += 1 # Increment iteration counter
#     # ... (rest of the old loop logic) ...
#     # Send the movement command        
#     # Continue moving with same values for 0.5 seconds
#     start_time = time.time()
#     while time.time() - start_time < 0.5:
#         # Print current movement values for debugging
#         print(f"Movement: yaw={yaw:.2f}, ascent={ascent:.2f}, roll={roll:.2f}, pitch={pitch:.2f}, person_found={person_found}")
#         drone.move(yaw, ascent, roll, pitch)
#         time.sleep(0.02) 
# if yolo_model is None:
#     print("Exiting program because YOLO model could not be loaded.")
# drone.land(True)