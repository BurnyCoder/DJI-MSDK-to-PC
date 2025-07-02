"""
Script to fly drone up 100 meters, take a picture, and land.
Inspired by fly_forward method from analyze-region-agent.py
"""

from OpenDJI import OpenDJI
import time
import os
from PIL import Image
import numpy as np

# IP address of the connected android device
IP_ADDR = os.environ.get("IP_ADDR", "100.105.85.101")

# Ascent speed (0.0 to 1.0, where 1.0 is maximum upward speed)
ASCENT_SPEED = 1  

# Estimated flight time to reach 100m (this may need adjustment based on actual drone speed)
# Assuming drone ascends at ~2-3 m/s at 0.5 speed, we need ~40-50 seconds
FLIGHT_DURATION_TO_100M = 5  # seconds


def capture_and_save_frame(drone, filename="altitude_photo.png"):
    """
    Captures a frame from the drone and saves it.
    
    Args:
        drone: The drone object
        filename: Name of the file to save
    
    Returns:
        True if successful, False otherwise
    """
    frame = drone.getFrame()
    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
        try:
            pil_image = Image.fromarray(frame.astype(np.uint8))
            pil_image.save(filename)
            print(f"Photo saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving photo: {e}")
            return False
    else:
        print("Could not capture frame.")
        return False


def fly_up_100m_and_photograph():
    """
    Main function to fly up 100 meters, take a photo, and land.
    """
    # Connect to the drone
    print(f"Connecting to drone at {IP_ADDR}...")
    
    with OpenDJI(IP_ADDR) as drone:
        try:
            # Enable SDK control
            print("Enabling SDK control...")
            result = drone.enableControl(True)
            print(f"Enable control response: {result}")
            time.sleep(2)
            
            # Take off
            print("Taking off...")
            result = drone.takeoff(True)
            print(f"Takeoff response: {result}")
            time.sleep(10)  # Wait for takeoff to complete and stabilize
            
            # Fly up to 100 meters
            print(f"Flying up to 100 meters (estimated {FLIGHT_DURATION_TO_100M} seconds)...")
            
            # Command interval (same as fly_forward method)
            command_interval = 0.02  # seconds (20ms)
            num_iterations = int(FLIGHT_DURATION_TO_100M / command_interval)
            
            start_time = time.time()
            for i in range(num_iterations):
                # Move up with specified ascent speed
                drone.move(pitch=0, yaw=0, roll=0, ascent=ASCENT_SPEED)
                time.sleep(command_interval)
                
                # Print progress every 1 second
                elapsed = time.time() - start_time
                if elapsed % 1 < command_interval:
                    print(f"Ascending... {elapsed:.1f}s elapsed")
            
            # Stop movement
            drone.move(0, 0, 0, 0)
            print("Reached target altitude (approximately 100m)")
            
            # Wait a moment to stabilize
            time.sleep(3)
            
            # Take a picture
            print("Taking picture at altitude...")
            if capture_and_save_frame(drone, f"photo_at_100m_{int(time.time())}.png"):
                print("Photo captured successfully!")
            else:
                print("Failed to capture photo, but continuing with landing...")
            
            # Wait a moment before descending
            time.sleep(3)
            
            # Fly down to a safe altitude before landing
            print(f"Descending back down (estimated {FLIGHT_DURATION_TO_100M} seconds)...")
            
            start_time = time.time()
            for i in range(num_iterations):
                # Move down with specified descent speed (negative ascent)
                drone.move(pitch=0, yaw=0, roll=0, ascent=-ASCENT_SPEED)
                time.sleep(command_interval)
                
                # Print progress every 1 second
                elapsed = time.time() - start_time
                if elapsed % 1 < command_interval:
                    print(f"Descending... {elapsed:.1f}s elapsed")
            
            # Stop movement
            drone.move(0, 0, 0, 0)
            print("Descended to lower altitude")
            
            # Wait a moment to stabilize before final landing
            time.sleep(2)
            
            # Land the drone
            print("Landing drone...")
            result = drone.land(True)
            print(f"Landing response: {result}")
            
            print("Mission complete!")
            
        except Exception as e:
            print(f"Error during flight: {e}")
            print("Attempting emergency landing...")
            try:
                drone.land(True)
            except:
                pass
            raise
        
        finally:
            # Ensure control is disabled
            try:
                drone.disableControl(True)
                print("SDK control disabled")
            except:
                pass


if __name__ == "__main__":
    print("=== Drone 100m Altitude Photo Mission ===")
    fly_up_100m_and_photograph()
    # print("WARNING: This script will fly the drone to 100 meters altitude!")
    # print("Ensure you have:")
    # print("- Clear airspace above")
    # print("- Legal permission to fly at this altitude")
    # print("- Good GPS signal")
    # print("- Sufficient battery")
    # print()
    
    # response = input("Type 'YES' to continue: ")
    # if response.upper() == 'YES':
    #     fly_up_100m_and_photograph()
    # else:
    #     print("Mission cancelled.")
