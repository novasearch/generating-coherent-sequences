import time
import os
import json
from tqdm import tqdm
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

RECIPES_PATH = '.json'
IMAGE_PATH = ''
CAPTION_RESULT_FILE = 'tmp.json'
MODEL_NAME = "blip2_vicuna_instruct"
MODEL_TYPE = 'vicuna13b'

NUM_RECIPES = 10

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# loads InstructBLIP model
model, vis_processors, _ = load_model_and_preprocess(name=MODEL_NAME, model_type=MODEL_TYPE, is_eval=True, device=device)

PROMPTS = {}


def prompt_no_context() -> str:
    # return "Given the steps, give a short description of the image. Say only what you see in the image. Be succint.\n"
    return "Given the steps, give a short description of the image. Do NOT make assumptions, say only what you see in the image.\n"



def construct_prompt(steps: str) -> str:
    # return f"Context:{N}{steps}{N}. Given the steps, what do we see in the image? Answer in a succint manner.{N}"
    # return f"Context:\n{steps}\n. Given the steps, what do we see in the image? When you mention objects, say their color.\n"
    return f"Steps:\n{steps}\n. {prompt_no_context()}"


final_results = {}
with open(CAPTION_RESULT_FILE, "r") as f:
    final_results = json.load(f)

# Timing variables
image_timings = []
total_start_time = time.perf_counter()

with open(RECIPES_PATH, 'r') as f:
    recipes = json.load(f)

recipe_items = list(recipes.items())[:NUM_RECIPES]

WINDOW = 3

for recipe_id, _recipe in tqdm(recipe_items):
    try:
        recipe = _recipe['recipe']
        last_step = ""
        
        context_arr = []

        try:
            steps = recipe['instructions']
            for i, step in enumerate(steps):
                current_step = f"Step {step['stepNumber']}: {step['stepText']}\n"
                # previous_steps += current_step
                image_id = f"{recipe_id}_{step['stepNumber']}"

                if len(context_arr) < WINDOW:
                    # context_arr.append(f"Step {step['stepNumber']}: {step['stepText']}\n")
                    context_arr.append(current_step)
                else:
                    print("DEBUG:", final_results)

                    context_arr[0] = f"Step 1: {final_results[f'{recipe_id}_{i-2+1}']['window']}\n"
                    context_arr[1] = f"Step 2: {steps[i-1]['stepText']}\n"
                    context_arr[2] = f"Step 3: {step['stepText']}\n"

                # load sample image
                raw_image = Image.open(os.path.join(IMAGE_PATH, f"{image_id}.png")).convert("RGB")

                # prepare the image
                image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

                # Timing for each image
                image_start_time = time.perf_counter()

                context = "".join(context_arr)

                print("Context:", context)

                prompt = construct_prompt(context)

                print("Prompt:", prompt)

                prompt_current_step = construct_prompt(current_step)

                prompt_last_step = construct_prompt(last_step + current_step)

                # Set last_step for next iteration
                last_step = current_step

                PROMPTS['no_context'] = prompt_no_context()
                PROMPTS['current_step'] = prompt_current_step
                PROMPTS['full_context'] = prompt
                PROMPTS['last_step'] = prompt_last_step

                PROMPTS['window'] = prompt

                results['step_description'] = current_step

                results = { prompt_id: model.generate({"image": image, "prompt": prompt}, max_length=77)[0] for prompt_id, prompt in PROMPTS.items() }

                print("#"*5, recipe_id, PROMPTS)
                print("*"*5, "Results:", results)

                final_results[image_id] = results

                # Timing for each image
                image_end_time = time.perf_counter()
                image_timings.append(image_end_time - image_start_time)
        except torch.cuda.OutOfMemoryError:
            print(f"No memory with {step['stepNumber']} steps. Skipping recipe.")
            continue
    except Exception as e: # catch all to make sure I can still dump the results
        print("Caught unexpected exception", e)
        continue


# Calculate total execution time
total_end_time = time.perf_counter()
total_execution_time = total_end_time - total_start_time

# Calculate average inference time per image
average_image_time = sum(image_timings) / len(image_timings)

print(f"Average inference time per image: {average_image_time:0.4f} s")
print(f"Total time for {len(recipe_items)} images: {total_execution_time}")

with open(CAPTION_RESULT_FILE, "w") as f:
    json.dump(final_results, f, indent=4)