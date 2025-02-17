import cv2
import base64
import json
import requests
import time

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Server URL (modify this to your HTTP endpoint)
    url = 'https://c82ywtbzth0ida-8080.dev-proxy.runpod.net'
    
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        print("captured frame")
        
        # Convert frame to jpg format
        _, buffer = cv2.imencode('.jpg', frame)
        print("converted frame")
        
        # Convert to base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        print("converted to base64")
        
        # Create JSON payload
        payload = json.dumps(img_base64)
        print("created payload")
        
        try:
            # Send HTTP POST request
            response = requests.post(url, json=payload)
            print(f"Received response: {response.text}")

            speed_and_throttle = json.loads(response.text)
            print(f"Speed and throttle: {speed_and_throttle}")
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            time.sleep(1)  # Wait a bit before retrying
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
