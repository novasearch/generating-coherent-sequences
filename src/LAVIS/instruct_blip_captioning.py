import time
import os
import json
from tqdm import tqdm
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

IMAGE_PATH = ''
CAPTION_RESULT_FILE = '.json'
MODEL_NAME = "blip2_vicuna_instruct"
MODEL_TYPE = 'vicuna13b'

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# loads InstructBLIP model
model, vis_processors, _ = load_model_and_preprocess(name=MODEL_NAME, model_type=MODEL_TYPE, is_eval=True, device=device)

PROMPTS = {
    "short_description": "Write a short description for the image.",
    "detailed_description": "Write a detailed description.",
    "what_see": "What can we see in the image?",
    "artificial": "What's in the image. Mention object properties and colors?",
    "artificial_longer": "What's in the image. Mention object properties, colors and materials?",
    "object_properties": "What are the objects in the image and properties, such as color?",
    "object_detector": "What are the objects in the image?",
    "object_detector_csv": "What are the objects in the image? Answer as a comma separated value list"
}

final_results = {}

# Timing variables
image_timings = []
total_start_time = time.perf_counter()

all_images = os.listdir(IMAGE_PATH)[:100]

for image_file in tqdm(all_images):
    image_id = os.path.splitext(image_file)[0]

    # load sample image
    raw_image = Image.open(os.path.join(IMAGE_PATH, image_file)).convert("RGB")

    # prepare the image
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    # Timing for each image
    image_start_time = time.perf_counter()

    results = { prompt_id: model.generate({"image": image, "prompt": prompt})[0] for prompt_id, prompt in PROMPTS.items() }

    final_results[image_id] = results

    # Timing for each image
    image_end_time = time.perf_counter()
    image_timings.append(image_end_time - image_start_time)

# Calculate total execution time
total_end_time = time.perf_counter()
total_execution_time = total_end_time - total_start_time

# Calculate average inference time per image
average_image_time = sum(image_timings) / len(image_timings)

print(f"Average inference time per image: {average_image_time:0.4f} s")
print(f"Total time for {len(all_images)} images: {total_execution_time}")

with open(CAPTION_RESULT_FILE, "w") as f:
    json.dump(final_results, f, indent=4)