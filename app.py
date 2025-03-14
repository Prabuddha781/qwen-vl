from flask import Flask, request, jsonify
import json
import logging

from hf import process_image

app = Flask(__name__)

@app.route('/', methods=['POST'])
def handle_request():
    try:
        # Receive image data from client
        image_data = request.data

        try:
            # Deserialize the image data
            image_payload = json.loads(image_data)
            image_payload = "data:image;base64," + image_payload
            results = process_image(image_payload)
            print("results", results)

            response = {
                "direction": results,
            }
            
            return jsonify(response)
            
        except json.JSONDecodeError:
            print("Invalid JSON received")
            return jsonify({"error": "Invalid JSON received"}), 400
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return jsonify({"error": str(e)}), 500
            
    except Exception as e:
        print(f"Error handling request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
