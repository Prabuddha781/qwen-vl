import cv2
import base64
import json
import websockets
import asyncio

async def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    # Connect to websocket server
    async with websockets.connect('wss://6u6kqcq33btrmw-8080.proxy.runpod.net', ping_interval=200, ping_timeout=100) as websocket:
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
            print("encoded base64")
            # Create JSON payload
            payload = json.dumps(img_base64)
            print("created payload")
            # Send to server
            await websocket.send(payload)
            print("sent payload")
            # Receive response
            response = await websocket.recv()
            print(f"Received response: {response}")
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # Clean up
                cap.release()
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    asyncio.run(main())
