import torch
import json
import random
from typing import List
from functools import partial
from visualization import display_images_with_text, display_tensor_grid, display_tensor_image, display_runs_side_by_side, display_images_and_tensors
from clip_similarity import text_text_similarity
from normalize import Normalizer
from latents import Latents
from diffuser import StableDiffusion
import numpy as np
from PIL import Image


def choose_iteration(similarity:float, max_step: int, threshold: float) -> int:
    return round(np.interp(similarity, [threshold, 1.0], [0, max_step]))


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


def real_images(recipe_id, steps):
    images = []
    size = (540, 540)

    for i, _ in enumerate(steps):
        image_id = f"{recipe_id}_{i+1}"
        image = Image.open(f".png")
        image = image.resize(size)
        images.append(image)

    return images


def generate_standard(steps, fixed_seed=False):
    generator = None
    if fixed_seed:
        # Setting a fixed seed for the whole recipe
        generator = torch.Generator("cuda").manual_seed(random.randint(0, 1000))

    images = []

    for step in steps:
        image = sd.call(prompt=step, generator=generator)

        images.append(image)

    return images


def generate_i2i(steps):
    images = []

    for i, step in enumerate(steps):
        if i == 0:
            image = sd.call(prompt=step)
        else:
            image = sd.call_i2i(prompt=step, image=images[i-1])

        images.append(image)

    return images


def generate_heuristics(generated_steps, threshold, recipe_id, fixed_goal=None, default_to_fixed_seed=True, real_steps=None):
    images = []
    latents_store_array = []
    stats = []

    # if we want a fixed seed we will create the generator here
    if default_to_fixed_seed:
        seed = random.randint(0, 100)
        print(f"Choosing {seed} as the seed for recipe {recipe_id}")
        generator = torch.Generator("cuda").manual_seed(seed)
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

                max_sim = np.max(sims)
                print(f"Max sim: {max_sim} (vs {threshold})")

                #if np.max(sims) > norm.normalize(threshold):
                if np.max(sims) > threshold:
                    # Pick step based on the highest similarity
                    recipe_step = np.argmax(sims)
                    # Map similarities to a noise iteration
                    step_goal = choose_iteration(np.max(sims), 3, threshold)

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

    if VISUALIZE and recipe_id and real_steps is not None:
        #display_tensor_grid(latents_store_array, 5, f"trash/{recipe_id}_tensor_grid")
        display_images_and_tensors(images, latents_store_array, real_steps, f"results/paper/tensors_{recipe_id}")

    return images#, stats

if __name__ == '__main__':
    DEBUG = False
    VISUALIZE = False

    REAL_RECIPES = "recipes_with_image_filtered.json"

    with open(REAL_RECIPES, "r") as f:
        real_recipes = json.load(f)

    STEPS_FILE = ".json"

    with open(STEPS_FILE, "r") as f:
        recipes = json.load(f)

    sd = StableDiffusion()
    sd.set_use_negative_prompts(True)
    #norm = Normalizer()

    #sd.load_i2i()

    KEYS_MAP = {
        # "Random Seed": "A",
        # "Fixed Seed": "B",
        # "Latent 1": "C",
        # "Latent 2": "D",
        # "Heuristic": "E",
        # "Image-to-Image": "F",
        # "Heuristic 70": "A",
        # "Heuristic 65": "B",
        # "Heuristic 60": "C",
        # "Heuristic 55": "D",
        # "Heuristic 50": "E",
        "Latent 1": "A",
        "Heuristic 50": "B"
    }

    NUM_RECIPES = 30

    recipes = list(recipes.items())[:min(NUM_RECIPES, len(recipes))]
    #recipes = recipes.items()

    keys = {}

    for recipe_id, recipe in recipes:
        print(recipe_id)
        images = []

        steps = recipe["steps_generated"]

        try:
            if random.choice([0, 1]):    
                images.append({"images": real_images(recipe_id, steps), "title": "A"})
                images.append({"images": generate_heuristics(steps, 0.50, recipe_id, fixed_goal=None, default_to_fixed_seed=True), "title": "B"})
                keys[recipe_id] = {"A": "Ground-Truth", "B": "Our Method"}
            else:
                images.append({"images": generate_heuristics(steps, 0.50, recipe_id, fixed_goal=None, default_to_fixed_seed=True), "title": "A"})
                images.append({"images": real_images(recipe_id, steps), "title": "B"})
                keys[recipe_id] = {"A": "Our Method", "B": "Ground-Truth"}
        except Exception as e:
            print(e)
            print("Most likely a file not found, we just go next recipe")
            print("I actually think this won't happen")
            continue

        display_runs_side_by_side(images, recipe["steps"], len(images), f"results/our_method_vs_real/{recipe_id}")

    with open("keys.json", "w") as f:
        json.dump(keys, f, indent=4)