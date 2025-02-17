from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import requests

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto", cache_dir="/workspace"
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

counter = 0

prompt = """You are a self-driving car. Your camera is situated on the top of your car and you are looking ahead. You need to stay strictly inside the pavement. If you are not centered on the pavement and the pavement is relatively straight, output one of SLIGHT_LEFT or SLIGHT_RIGHT so that you can steer back to the center. If you are at an intersection, output HEAVY_LEFT, HEAVY_RIGHT, SLIGHT_LEFT or SLIGHT_RIGHT to turn. If you are already centered on the pavement, output STRAIGHT.

ONLY output one of these values: HEAVY_LEFT, HEAVY_RIGHT, SLIGHT_LEFT, SLIGHT_RIGHT, STRAIGHT and in upper casing all the time."""

try:
    response = requests.get("https://raw.githubusercontent.com/Prabuddha781/qwen-vl/main/prompt")
    if response.status_code == 200 and response.text != prompt:
        print("updating prompt. new prompt: ", response.text)
        prompt = response.text
except requests.RequestException as e:
    print(f"Error fetching prompt: {e}")


def process_image(image):
    # Create message structure for the model
    global counter
    global prompt
    counter += 1

    if counter % 30 == 0:
        try:
            response = requests.get("https://raw.githubusercontent.com/Prabuddha781/qwen-vl/main/prompt")
            if response.status_code == 200 and response.text != prompt:
                print("updating prompt. new prompt: ", response.text)
                prompt = response.text
        except requests.RequestException as e:
            print(f"Error fetching prompt: {e}")

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
                }
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
        # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]  # Get first (and only) result
    return output_text