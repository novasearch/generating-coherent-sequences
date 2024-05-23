import torch
from tqdm import tqdm
import json
import random
from typing import List
from functools import partial
from visualization import display_images_with_text, display_tensor_grid, display_tensor_image, display_runs_side_by_side, display_images_and_tensors
from clip_similarity import text_text_similarity
from normalize import Normalizer
from latents import Latents
from diffuser import StableDiffusion
import torch.nn as nn
import numpy as np


def choose_iteration(similarity:float, max_step: int, threshold: float) -> int:
    return round(np.interp(similarity, [threshold, 1.0],[0, max_step]))


def compute_similarities(text: str, texts: List[str]) -> List[float]:
    return [text_text_similarity(text, t) for t in texts]


def capture_latents(latents_store: Latents, step: int, timestep: int, latents: torch.FloatTensor):
    if DEBUG:
        print("Inside callback")
        print("Step:", step)
        print("Timestep:", timestep)
        print("Latents Shape:", latents.shape)

    latents_store.add_latents(latents)

    return

def generate_with_latents(generated_steps, threshold, recipe_id, fixed_goal=None, default_to_fixed_seed=True):
    images = []
    latents_store_array = []
    stats = []

    # if we want a fixed seed we will create the generator here
    if default_to_fixed_seed:
        generator = torch.Generator("cuda").manual_seed(int(recipe_id))
    else:
        # Explicit for readability
        generator = None        

    for i, step in enumerate(generated_steps):
        latents_store = Latents()

        previous_latents = None

        if i != 0:
            # If there is no goal, we use the heuristic to calculate it  
            if not fixed_goal:
                # Compute similarities
                sims = compute_similarities(step, generated_steps[:i])
                #sims = [norm.normalize(sim) for sim in sims]

                print(f"Recipe: {recipe_id}\nSimilarities: {sims}")

                #if np.max(sims) > norm.normalize(SIMILARITY_THRESHOLD):
                if np.max(sims) > threshold:
                    # Pick step based on the highest similarity
                    recipe_step = np.argmax(sims)
                    # Map similarities to a noise iteration
                    step_goal = choose_iteration(np.max(sims), 3)

                    print(f"Choosing Step: {recipe_step}\nChoosing Iteration: {step_goal}\n")

                    stats.append({"step": int(recipe_step), "latent": int(step_goal)})

                    previous_latents = latents_store_array[recipe_step][step_goal]
                else:
                    previous_latents = None
                    print("Choosing Step: -1\nChoosing Iteration: -1\n")
                    stats.append({"step": -1, "latent": -1})
            else:
                print("Fixed goal is set")
                previous_latents = latents_store_array[i-1][fixed_goal]
                stats.append({"step": i-1, "latent": fixed_goal})
        else:
            print("First iteration")
            stats.append({"step": -1, "latent": -1})

        f = partial(capture_latents, latents_store)

        """
            Not sure how generator and latents interact so I'm making the calls separate for safety.
        """
        if default_to_fixed_seed and previous_latents is None:
            assert generator
            print("Defaulting to fixed seed")
            image = sd.call(step, generator=generator, callback=f, callback_steps=1)
        else:
            image = sd.call(step, latents=previous_latents, callback=f, callback_steps=1)

        images.append(image)

        latents_store_array.append(latents_store.get_history())

    if VISUALIZE and recipe_id:
        #display_tensor_grid(latents_store_array, 5, f"trash/{recipe_id}_tensor_grid")
        display_images_and_tensors(images, latents_store_array, f"results/paper/tensors_{recipe_id}")

    return images, stats

if __name__ == '__main__':
    DEBUG = False
    VISUALIZE = False
    SIMILARITY_THRESHOLD = 0.50

    STEPS_FILE = ".json"

    result_file = ".json"
    print(result_file)

    with open(STEPS_FILE, "r") as f:
        recipes = json.load(f)

    sd = StableDiffusion()
    sd.set_use_negative_prompts(True)

    #norm = Normalizer()

    NUM_RECIPES = 5

    recipes = list(recipes.items())[:min(NUM_RECIPES, len(recipes))]

    stats_ds = {}

    for recipe_id, recipe in recipes:
        real_steps = recipe["steps"]
        generated_steps = recipe["steps_generated"]

        imgs, stats = generate_with_latents(generated_steps, SIMILARITY_THRESHOLD, recipe_id, fixed_goal=None, default_to_fixed_seed=True)

        stats_ds[recipe_id] = stats

        display_runs_side_by_side([{"images": imgs, "title": "Images"}], recipe["steps"], len([imgs]), f"results/trash/{recipe_id}")

    with open(result_file, "w") as f:
        json.dump(stats_ds, f, indent=4)
