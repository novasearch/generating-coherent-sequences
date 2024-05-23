import json
from generate_images import gen_images
import sys
import os


def split_prompts(prompts, worker_id, num_workers):
    total_prompts = len(prompts)
    prompts_per_program = total_prompts // num_workers
    start_idx = prompts_per_program * worker_id
    end_idx = start_idx + prompts_per_program if worker_id < num_workers - 1 else total_prompts

    return prompts[start_idx:end_idx]


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: python main.py <prompts_file> <image_folder> <prompt_type> <worker_id> <num_workers>")
        exit(1)

    prompts = sys.argv[1]
    images_dir = sys.argv[2]
    prompt_type = sys.argv[3]

    # If dir doesn't exist, create it
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Prompts[Prompt{"id": id, "prompt": step_text}]
    with open(prompts) as f:
        prompts = json.load(f)

    worker_id = int(sys.argv[4]) # Worker id
    num_workers = int(sys.argv[5]) # Total number of workers

    worker_prompts = split_prompts(prompts, worker_id, num_workers)

    gen_images(worker_prompts, images_folder=images_dir, prompt_type=prompt_type, num_images_per_prompt=1, num_images_per_batch=1, upload=True)


# previous_prompt_id = None

# for recipe_id, recipe in recipes.items():
#     print(recipe_id, recipe)
#     for step in recipe['instructions']:
#         previous_prompt_id = None
#         step_id = step['step_id']
#         prompt_id = f"{recipe_id}_{step_id}"
#         if step_id > 1:
#             previous_prompt_id = f"{recipe_id}_{step_id - 1}"

#         print(f"Doing recipe {prompt_id} . Previous id: {previous_prompt_id}")
            
#         prompt = {"id": prompt_id, "prompt": prompts[prompt_id]}

#         gen_image_to_image(prompt, previous_prompt_id, images_folder, 4)