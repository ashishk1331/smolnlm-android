from flask import Flask, request, jsonify
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import io
import base64
import os
import re

app = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, _attn_implementation="eager"
).to(DEVICE)  # Or "cpu" if no GPU

@app.route('/', methods=['GET'])  # Health check route
def health_check():
    return jsonify({'status': 'ok', 'message': 'Server is up and running!'})

def process_image_from_base64(base64_string, temp_dir="temp_images"):
    """Decodes base64 and saves to a temporary file, returning the path."""
    try:
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))

        w, h = image.width, image.height
        new_w = 512

        if w > 600:
            image = image.resize((new_w, round(h * (w/new_w))))

        temp_path = os.path.join(temp_dir, "temp_image.png")  # Or .jpg, etc.
        image.save(temp_path)  # Save the image temporarily

        return temp_path  # Return the file path
    except Exception as e:
        print(f"Error processing image: {e}")
        return None



@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_base64 = data['image'] # Get base64 image
    text = data['text']

    temp_image_path = process_image_from_base64(image_base64) # Get path to image file

    if temp_image_path is None:
        return jsonify({'error': 'Error processing image'})

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},  # Use the text from the request
                {"type": "image", "path": temp_image_path},  # Use the temporary path
            ]
        },
    ]

    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        result = re.sub(r"User:.*Assistant:\s*", "", generated_texts[0], flags=re.DOTALL)

        # Cleanup: Remove the temporary image file
        os.remove(temp_image_path) # Important to prevent disk filling up

        return jsonify({'result': result})

    except Exception as e:
        # Cleanup even if there is an error
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000) # Run on localhost:5000