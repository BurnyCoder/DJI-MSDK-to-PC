"""
This application flies a DJI drone forward for 5 seconds, capturing one frame per second.
It then sends these frames to OpenAI for analysis.
"""

from OpenDJI import OpenDJI
import time
import os
import base64
from io import BytesIO
from openai import OpenAI
from PIL import Image
import numpy as np

# IP address of the connected android device
# IP_ADDR = os.environ.get("IP_ADDR", "100.93.47.145")
IP_ADDR = os.environ.get("IP_ADDR", "100.85.47.22")

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

# Movement factor
MOVE_VALUE = float(os.environ.get("MOVE_VALUE", "0.1"))


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
        print("Takeoff complete.")

        print("Starting 5-second forward flight and capturing frames.")
        captured_frames = []
        
        # Start moving forward
        drone.move(pitch=MOVE_VALUE, yaw=0, roll=0, ascent=0)

        # Capture frames for 5 seconds, one per second
        for i in range(5):
            time.sleep(1) # Wait for 1 second
            frame = drone.getFrame()
            if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                captured_frames.append(frame)
                print(f"Frame {i+1} captured.")
            else:
                print(f"Could not capture frame {i+1}.")
        
        # Stop the drone
        drone.move(0, 0, 0, 0)
        print(f"Forward flight complete. Captured {len(captured_frames)} frames.")

        if captured_frames:
            print("Sending frames to OpenAI for analysis...")
            
            base64_frames = []
            for frame in captured_frames:
                pil_image = Image.fromarray(frame.astype(np.uint8))
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                base64_frames.append(img_base64)

            prompt_messages = [
                {
                    "type": "text",
                    "text": "Analyze these frames from a drone's 5-second forward flight. Describe what you see, including any objects, people, or significant features in the environment. Note any changes or movement across the frames.",
                }
            ]
            for b64_frame in base64_frames:
                prompt_messages.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_frame}"}
                    }
                )

            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_messages,
                        }
                    ],
                    max_tokens=500,
                )
                analysis_result = response.choices[0].message.content
                print("\n--- OpenAI Analysis ---")
                print(analysis_result)
                print("-----------------------\n")

                # Write the analysis to a file
                with open("analysis.txt", "w", encoding="utf-8") as f:
                    f.write(analysis_result)
                print("Analysis saved to analysis.txt")
            except Exception as e:
                print(f"Error during OpenAI API call: {e}")


    print("Landing drone...")
    drone.land(True)
    print("Drone landed. Exiting program.")