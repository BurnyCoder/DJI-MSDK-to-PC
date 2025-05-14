from OpenDJI import OpenDJI

import cv2
import numpy as np
import keyboard

"""
In this example you will see how to use the action command.
The action command is very similar to the set command,
but it is more directed to physical actions then setting parameters.
In this example you will control the camera (more precisely the gimbal),
and set it view direction. Make note on the output from the terminal!

    Press X - to close the program.

    Press A/D - to move the camera left / right (yaw)
    Press W/S - to move the camera up / down (pitch)
    Press Q/E - to tilt the camera to the left / right (roll)
"""

# IP address of the connected android device
import os
IP_ADDR = os.getenv("DRONE_IP_ADDR", "192.168.1.115")

# The image from the drone can be quit big,
#  use this to scale down the image:
SCALE_FACTOR = 0.25

# Create blank frame
BLANK_FRAME = np.zeros((1080, 1920, 3))
BLNAK_FRAME = cv2.putText(BLANK_FRAME, "No Image", (200, 300),
                          cv2.FONT_HERSHEY_PLAIN, 10,
                          (255, 255, 255), 10)

# Control angle
ANGLE_STEP = 0.3
pitch = 0.0
roll = 0.0
yaw = 0.0

import uuid
import time

# Connect to the drone
with OpenDJI(IP_ADDR) as drone:

    
    log_file_name = f"yt_realtime_tracking_log_{uuid.uuid4().hex[:8]}.txt" # Specific log for this run

    # Create a directory for saving frames for this run
    base_log_name = os.path.splitext(os.path.basename(log_file_name))[0]
    frames_dir = os.path.join(os.path.dirname(log_file_name) or "logs", base_log_name + "_frames")
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    print(f"Frames for this run will be saved in: {frames_dir} (YT_RT)")
    
    print("Initiating YOLO-based REAL-TIME automated person tracking sequence (YT_RT)...")
    global yt_drone_connection, yt_yolo_model # Referencing this module's globals

    result = drone.enableControl(True)
    print(f"Enable SDK control command sent (YT_RT). Drone response: {result}")
    
    print("Sending takeoff command (YT_RT)...")
    takeoff_result = drone.takeoff(True)
    print(f"Takeoff result (YT_RT): {takeoff_result}")

    print("Stabilizing after takeoff (YT_RT)...")
    time.sleep(9.5) # Keep stabilization period

    print(f"Starting.")
    

    # Press 'x' to close the program
    print("Press 'x' to close the program")
    while cv2.waitKey(20) != ord('x'):

        # Get frame
        frame = drone.getFrame()

        # Control with the keyboard the gimbal orientation
        if keyboard.is_pressed("a"): yaw -= ANGLE_STEP
        if keyboard.is_pressed("d"): yaw += ANGLE_STEP
        if keyboard.is_pressed("q"): roll -= ANGLE_STEP
        if keyboard.is_pressed("e"): roll += ANGLE_STEP
        if keyboard.is_pressed("s"): pitch -= ANGLE_STEP
        if keyboard.is_pressed("w"): pitch += ANGLE_STEP
        
        # The command to control the gimbel,
        # Why it look like that ? thats what DJI designed.
        # How to know what other commands look like ?
        # send on the query server the follwing command:
        #   'help Gimbal RotateByAngle'
        # you can use the QueryExampleRaw, or type here the command:
        #   print(drone.getKeyInfo("Gimbal", "RotateByAngle"))
        command_argument = ('{'
            '"mode":65535,'
            f'"pitch":{pitch:5},'
            f'"roll":{roll:5},'
            f'"yaw":{yaw:5},'
            '"pitchIgnored":false,'
            '"rollIgnored":false,'
            '"yawIgnored":false,'
            '"duration":0,'
            '"jointReferenceUsed":false,'
            '"timeout":10'
        '}')
        
        # Send the action and print the result,
        # Note, some action does not need value to performe, for example:
        #   drone.action("RemoteController", "RebootDevice")
        print(drone.action(OpenDJI.MODULE_GIMBAL, "RotateByAngle", command_argument))

        # What to do when no frame available
        if frame is None:
            frame = BLANK_FRAME
    
        # Resize frame - optional
        frame = cv2.resize(frame, dsize = None,
                           fx = SCALE_FACTOR,
                           fy = SCALE_FACTOR)
        
        # Show frame
        cv2.imshow("Live video", frame)