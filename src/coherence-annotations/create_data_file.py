import json
import csv
import os
import sys

BUCKET_NAME = ""

HEADER = ["recipe_name", "method", "image_id", "previous_text", "previous_image_url", "text", "image_url"]


def get_url(image_path):
    return f"https://{BUCKET_NAME}.s3.amazonaws.com/{image_path}.png"


def create_file(steps_file, result_file, prompt_type, count=-1):
    
    # We want the keys from generated_steps_file
    with open(steps_file, "r") as f:
        recipes = json.load(f)

    recipes = list(recipes.items())[:min(count, len(recipes))]

    with open(result_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        for recipe_id, recipe in recipes:
            recipe_name = recipe["recipe_name"]
            steps = recipe["steps"]

            previous_step = steps[0]
            previous_url = get_url(f"{prompt_type}/{recipe_id}_1")

            for i, step in enumerate(steps[1:]):
                # i+2 since we start at step 1
                image_id = f"{recipe_id}_{i+2}"
                url = get_url(f"{prompt_type}/{image_id}")
                row = [recipe_name, prompt_type, image_id, previous_step, previous_url, step, url]
                writer.writerow(row)
                previous_step = step
                previous_url = url

if __name__ == "__main__":
    STEPS_FILE = '.json'
    prompt_type = "latent_1"
    RESULTS_DIR = f""
    CSV_RESULT_FILE = os.path.join(RESULTS_DIR, f"{prompt_type}.csv")

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    create_file(STEPS_FILE, CSV_RESULT_FILE, prompt_type, count=20)