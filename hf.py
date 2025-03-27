from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import requests
import time
import torch

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto", cache_dir="/home/pierre/hf_hub"
)
# Set model to evaluation/inference mode
model.eval()

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")

counter = 0

prompt = """You love to manipulate robots. The input resolution for the image you are looking at is 1024x1024. Your task right now is to pick up the red bin. Output the point in the bin where a human would put their hands on to grasp the bin as a tuple (x, y). Output the next five points in space the robot's right gripper (colored in pink) would have to go to pick up the bin. The first point should be the point where your right gripper's head is, ending at the red bin's grasp point with the five points evenly spaced apart as a list of tuples, where each tuple is a point (x, y). Also output the bounding box of the red bin as a tuple (x1, y1, x2, y2). Give your output as a map with the following keys: 'grasp', 'trajectory', 'bounding_box'."""

def process_image(image):
    # Create message structure for the model
    global counter
    global prompt
    counter += 1

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": image
                },
                {
                    "type": "text",
                    "text": prompt
                },
                # {
                #     "type": "image",
                #     "image": gripper_image,
                # }
            ]
        }
    ]
    
    # Prepare inputs for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Generate output with inference mode
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=500)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]  # Get first (and only) result
    return output_text

# Example usage: Call process_image with an image from the internet
if __name__ == "__main__":
    # Download an image from the internet
    image_url = "https://miro.medium.com/v2/resize:fit:1400/0*sWqDN0zIIOU_EW3l.jpg"
    response = requests.get(image_url, stream=True)
    response.raise_for_status()
    
    # Process the image with PIL
    from PIL import Image
    from io import BytesIO
    
    image = Image.open("robot_arm.jpg")
    image = image.resize((1024, 1024))

    # gripper_image = Image.open("gripper.png")   
    # gripper_image = gripper_image.resize((248, 248))

    start_time = time.time()
    # Call the process_image function
    result = process_image(image)
    end_time = time.time()
    print("Model output:", result)
    print("Time taken:", end_time - start_time)
