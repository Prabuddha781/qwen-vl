from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto", cache_dir="/workspace"
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def process_image(image):
    # Create message structure for the model
    print("received image")
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
                    "text": "What direction should I steer based on this road image? Answer with exactly one word: LEFT, RIGHT, or STRAIGHT"
                }
            ]
        }
    ]
    
    # Prepare inputs for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print("processed text")
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    print("inputs")
    # Generate output
    # print(inputs)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    print("generated ids top")
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    print("generated ids")
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]  # Get first (and only) result
    print("output text")
    return output_text