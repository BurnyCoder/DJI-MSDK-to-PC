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
# from ultralytics import YOLO
import time
import os

import keyboard
import cv2
import numpy as np
import threading
import base64
from io import BytesIO
from openai import OpenAI
from PIL import Image

# IP address of the connected android device
# IP_ADDR = os.environ.get("IP_ADDR", "192.168.1.115")
IP_ADDR = os.environ.get("IP_ADDR", "100.124.105.254")


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
# try:
#     yolo_model = YOLO("yolov8n.pt")
#     print("YOLO model loaded successfully.")
# except Exception as e:
#     print(f"Error loading YOLO model: {e}")
#     print("Please ensure 'yolov8n.pt' is accessible and ultralytics is installed.")
#     yolo_model = None

# --- Shared data between threads ---
shared_data_lock = threading.Lock()
latest_movement_commands = {
    "yaw": 0.0,
    "ascent": 0.0,
    "roll": 0.0,
    "pitch": 0.0,
    "person_found": False # Added for logging in movement thread
}
stop_threads_event = threading.Event()
# --- End of Shared data ---

# --- Thread 1: Frame Processing and YOLO ---
def frame_processing_and_yolo_thread_func(drone_obj, blank_frame_img, frames_dir_path):
    print("Processing thread started.")
    iter_count = 0 # Initialize iteration counter for this thread
    person_sighted_in_previous_iteration = False
    consecutive_no_person_scans = 0

    while not stop_threads_event.is_set():
        start_processing_time = time.time()
        iter_count += 1

        current_frame_data = drone_obj.getFrame()
        frame_for_processing = blank_frame_img.copy() if current_frame_data is None else current_frame_data.copy()

        # Local movement variables for this cycle's calculation
        calculated_yaw = 0.0
        calculated_ascent = 0.0 # Assuming ascent and roll are not dynamically changed by YOLO yet
        calculated_roll = 0.0
        calculated_pitch = 0.0
        person_detected_this_frame = False

        if openai_client and not np.array_equal(frame_for_processing, blank_frame_img) and frame_for_processing.shape[0] > 0 and frame_for_processing.shape[1] > 0:
            pil_image = Image.fromarray(frame_for_processing.astype(np.uint8))
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            if person_sighted_in_previous_iteration:
                state_context = "A person was visible in the last frame. Continue tracking."
            elif consecutive_no_person_scans > 0:
                state_context = f"A person was visible previously but has been lost for {consecutive_no_person_scans} frame(s). Decide whether to scan or hold position."
            else:
                state_context = "No person has been sighted recently. Scan for a person or hold position."

            prompt = f"""Analyze this image from a drone's camera. Context: {state_context}
Instructions:
1. Determine if a person is clearly visible (Yes/No).
2. Provide drone movement commands (`rotation` for rotation, `move` for forward/backward).
   - If Visible: Yes, provide `rotation` to turn towards the person and `move` (positive value, 0.0 to 1.0) to move forward.
   - If Visible: No, provide `rotation` (non-zero) to scan for the person. Keep `move` at 0.0.
   - `rotation` range: -1.0 (rotate left) to 1.0 (rotate right).
   - `move` range: 0.0 to 1.0 (only forward movement considered).
Response Format:
Respond ONLY in the format: "Visible: [Yes/No], rotation: [float], move: [float]"

Examples:
- Person centered, move forward: "Visible: Yes, rotation: 0.0, move: 0.5"
- Person slightly right, move forward: "Visible: Yes, rotation: 0.1, move: 0.3"
- No person, scan right: "Visible: No, rotation: -0.5, move: 0.0"
- No person, scan left: "Visible: No, rotation: 0.5, move: 0.0" """

            try:
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

                parts = llm_output.split(',')
                visible_part = parts[0].split(':')[1].strip().lower()
                rotation_part = parts[1].split(':')[1].strip()
                move_part = parts[2].split(':')[1].strip()

                if visible_part == 'yes':
                    person_detected_this_frame = True
                
                calculated_yaw = max(-1.0, min(1.0, float(rotation_part)))
                calculated_pitch = max(0.0, min(1.0, float(move_part)))

            except Exception as e:
                print(f"Error parsing OpenAI response or in API call: {e}")
                person_detected_this_frame = False
                calculated_yaw = 0.0
                calculated_pitch = 0.0

            if person_detected_this_frame:
                person_sighted_in_previous_iteration = True
                consecutive_no_person_scans = 0
            else:
                if person_sighted_in_previous_iteration:
                    consecutive_no_person_scans = 1
                elif consecutive_no_person_scans > 0:
                    consecutive_no_person_scans += 1
                person_sighted_in_previous_iteration = False
        else:
            calculated_yaw = 0.0
            calculated_pitch = 0.0
        
        # Update shared movement commands
        with shared_data_lock:
            latest_movement_commands["yaw"] = calculated_yaw
            latest_movement_commands["ascent"] = calculated_ascent
            latest_movement_commands["roll"] = calculated_roll
            latest_movement_commands["pitch"] = calculated_pitch
            latest_movement_commands["person_found"] = person_detected_this_frame

        # Image Saving Logic (using calculated values before lock or from local vars)
        if not np.array_equal(frame_for_processing, blank_frame_img) and frame_for_processing.shape[0] > 0 and frame_for_processing.shape[1] > 0:
            yaw_str = f"{calculated_yaw:.2f}".replace('.', 'p').replace('-', 'neg')
            pitch_str = f"{calculated_pitch:.2f}".replace('.', 'p').replace('-', 'neg')
            person_str = "T" if person_detected_this_frame else "F"
            
            detailed_frame_filename = f"yaw{yaw_str}_pitch{pitch_str}_person{person_str}_iter{iter_count}.jpg"
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
        local_yaw, local_ascent, local_roll, local_pitch, local_person_found = 0.0, 0.0, 0.0, 0.0, False
        with shared_data_lock:
            local_yaw = latest_movement_commands["yaw"]
            local_ascent = latest_movement_commands["ascent"]
            local_roll = latest_movement_commands["roll"]
            local_pitch = latest_movement_commands["pitch"]
            local_person_found = latest_movement_commands["person_found"] # Get for logging

        # Print current movement values being sent to the drone
        print(f"Movement: yaw={local_yaw:.2f}, ascent={local_ascent:.2f}, roll={local_roll:.2f}, pitch={local_pitch:.2f}, person_found={local_person_found}")
        drone_obj.move(local_yaw, local_ascent, local_roll, local_pitch)
        time.sleep(MOVEMENT_COMMAND_INTERVAL) # Send command every 0.02 seconds
    print("Movement thread stopped.")

# Connect to the drone
with OpenDJI(IP_ADDR) as drone:
    if openai_client is None:
        print("OpenAI client not initialized. Exiting application.")
    else:
        result = drone.enableControl(True)
        print(f"Enable SDK control command sent. Drone response: {result}")

        print("Taking off...")
        drone.takeoff(True)
        time.sleep(10)
        print("Taking off complete")

        # Create and start threads
        processing_thread = threading.Thread(target=frame_processing_and_yolo_thread_func, 
                                             args=(drone, BLANK_FRAME, FRAMES_DIR))
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