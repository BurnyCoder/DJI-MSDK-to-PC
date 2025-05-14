from OpenDJI import OpenDJI
from ultralytics import YOLO
import time
import os

import keyboard
import cv2
import numpy as np

"""
In this example you can fly and see video from the drone in live!
Like a computer game, move the drone with the keyboard and see its image
on your computer screen!

    press F - to takeoff the drone.
    press R - to land the drone.
    press E - to enable control from keyboard (joystick disabled)
    press Q - to disable control from keyboard (joystick enabled)
    press X - to close the problam

    YOLO-based movement:
    - The drone will attempt to find a 'person' in the video feed.
    - If a person is detected, it will rotate to center them.
    - If a person is centered, it will move forward.
    - If no person is detected, it will rotate to scan.
"""

# IP address of the connected android device
IP_ADDR = "192.168.1.115"

# The image from the drone can be quit big,
#  use this to scale down the image:
SCALE_FACTOR = 0.5

# Movement factors
MOVE_VALUE = 0.015
ROTATE_VALUE = 0.15
CENTER_THRESHOLD_PERCENT = 0.1

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

# Connect to the drone
with OpenDJI(IP_ADDR) as drone:

    result = drone.enableControl(True)
    print(f"Enable SDK control command sent. Drone response: {result}")

    print("Taking off...")
    drone.takeoff(True)
    time.sleep(10)
    print("Taking off complete")

    iter_count = 0 # Initialize iteration counter

    # Press 'x' to close the program
    print("Press 'x' to close the program")
    while not keyboard.is_pressed('x') and yolo_model is not None:
        iter_count += 1 # Increment iteration counter

        # Show image from the drone
        # Get frame
        current_frame_data = drone.getFrame()

        # What to do when no frame available
        if current_frame_data is None:
            # Use a copy to avoid modifying the global BLANK_FRAME if it were mutable
            # and to ensure 'frame_for_processing' is always a new object for this iteration
            frame_for_processing = BLANK_FRAME.copy() 
        else:
            # Work with a copy of the received frame for safety
            frame_for_processing = current_frame_data.copy()

        # Resize frame for display
        # Create a display copy from the frame that will be processed/saved
        frame_for_display = frame_for_processing.copy()
        if not np.array_equal(frame_for_processing, BLANK_FRAME): # If it's a real frame
            try:
                if frame_for_processing.shape[0] > 0 and frame_for_processing.shape[1] > 0:
                     frame_for_display = cv2.resize(frame_for_processing, dsize=None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            except cv2.error as e:
                print(f"Error resizing frame for display: {e}")
                # Fallback to displaying a scaled blank frame if resize fails
                frame_for_display = cv2.resize(BLANK_FRAME.copy(), dsize=None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        else: # If frame_for_processing was BLANK_FRAME to begin with
            frame_for_display = cv2.resize(BLANK_FRAME.copy(), dsize=None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)

        # Show frame
        cv2.imshow("Live video", frame_for_display)
        cv2.waitKey(20)
        
        # Movement variables
        yaw = 0.0
        ascent = 0.0
        roll = 0.0
        pitch = 0.0
        dx_val = None  # Initialize dx_val for filename and logic
        person_found = False # Reset for current frame's analysis

        # YOLO-based movement logic
        # 'frame_for_processing' here is the original unscaled frame (or a copy of BLANK_FRAME)
        if not np.array_equal(frame_for_processing, BLANK_FRAME) and frame_for_processing.shape[0] > 0 and frame_for_processing.shape[1] > 0:
            # This is the frame YOLO will process
            
            H, W, _ = frame_for_processing.shape
            frame_center_x = W / 2.0
            current_center_threshold_pixels = W * CENTER_THRESHOLD_PERCENT

            yolo_results = yolo_model(frame_for_processing, verbose=False)
            # person_found is already initialized to False for this iteration

            if yolo_results and yolo_results[0].boxes:
                for box in yolo_results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = yolo_model.names[class_id]
                    
                    if class_name == 'person':
                        person_coords = box.xyxy[0].cpu().numpy() 
                        px1, _, px2, _ = person_coords
                        person_cx = (px1 + px2) / 2.0
                        dx_val = person_cx - frame_center_x # Assign to dx_val

                        if abs(dx_val) > current_center_threshold_pixels:
                            yaw = ROTATE_VALUE if dx_val > 0 else -ROTATE_VALUE
                            pitch = 0.0 # Stop forward movement while rotating
                        else: # Person is centered
                            yaw = 0.0 # Stop rotation
                            pitch = MOVE_VALUE # Move forward
                        
                        person_found = True # Set person_found flag
                        break # Found a person, no need to check other boxes

            if not person_found: # If no person was found after checking all boxes
                yaw = ROTATE_VALUE # Scan
                pitch = 0.0
        else: # This 'else' corresponds to (frame_for_processing is BLANK_FRAME or has invalid shape)
            # No YOLO analysis, so no person found by YOLO this iteration
            # person_found remains False (as initialized)
            yaw = 0.0 # Default to no movement
            pitch = 0.0
            # ascent and roll are already 0.0
            # dx_val remains None
        
        # --- Image Saving Logic ---
        # Save 'frame_for_processing' if it's not the BLANK_FRAME and is valid
        if not np.array_equal(frame_for_processing, BLANK_FRAME) and frame_for_processing.shape[0] > 0 and frame_for_processing.shape[1] > 0:
            yaw_str = f"{yaw:.2f}".replace('.', 'p').replace('-', 'neg')
            pitch_str = f"{pitch:.2f}".replace('.', 'p').replace('-', 'neg')
            person_str = "T" if person_found else "F"
            
            dx_for_filename = "NA"
            if person_found and dx_val is not None: # Check person_found status
                dx_for_filename = f"{dx_val:.1f}".replace('.', 'p').replace('-', 'neg')

            detailed_frame_filename = f"yaw{yaw_str}_pitch{pitch_str}_person{person_str}_dx{dx_for_filename}_iter{iter_count}.jpg"
            frame_saved_path = os.path.join(FRAMES_DIR, detailed_frame_filename)
            
            try:
                cv2.imwrite(frame_saved_path, frame_for_processing) 
                # print(f"Saved frame: {frame_saved_path}") # Uncomment for verbose logging
            except Exception as e_save:
                print(f"Error saving frame {frame_saved_path}: {e_save}")
        # --- End of Image Saving Logic ---

        # Send the movement command        
        # Continue moving with same values for 0.5 seconds
        start_time = time.time()
        while time.time() - start_time < 0.5:
            # Print current movement values for debugging
            print(f"Movement: yaw={yaw:.2f}, ascent={ascent:.2f}, roll={roll:.2f}, pitch={pitch:.2f}, person_found={person_found}")
            drone.move(yaw, ascent, roll, pitch)
            time.sleep(0.02)  # Small sleep to prevent CPU overload

    if yolo_model is None:
        print("Exiting program because YOLO model could not be loaded.")
    
    drone.land(True)
    cv2.destroyAllWindows()