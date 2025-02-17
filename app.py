import websockets
import asyncio
import json

from hf import process_image

SMALL_FORWARD = 410
STEERING_LEFT = 350
STEERING_LEFT_MORE = 330
STEERING_MIDDLE = 375
STEERING_RIGHT = 400
STEERING_RIGHT_MORE = 420

async def handle_websocket(websocket):
    try:
        while True:
            # Receive image data from client
            image_data = await websocket.recv()

            try:
                # Deserialize the image data
                image_payload = json.loads(image_data)
                image_payload = "data:image;base64," + image_payload
                results = process_image(image_payload)
                print("results", results)
                # For now using dummy values
                steering = STEERING_MIDDLE
                throttle = SMALL_FORWARD

                response = {
                    "steering": steering,
                    "throttle": throttle
                }
                
                await websocket.send(json.dumps(response))
                
            except json.JSONDecodeError:
                print("Invalid JSON received")
                continue
            except Exception as e:
                print(f"Error processing request: {str(e)}")
                continue
                
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def start_server():
    server = await websockets.serve(handle_websocket, "0.0.0.0", 8080)
    print("WebSocket server started on ws://0.0.0.0:8080")
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.get_event_loop().run_until_complete(start_server())
    except KeyboardInterrupt:
        print("\nStopping drive commands...")
