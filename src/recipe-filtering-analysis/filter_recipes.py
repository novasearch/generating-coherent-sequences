import json
from analyze_recipes import average_number_steps
import re

def filter_recipe_num_steps(recipes, target_steps=5, window=1):
    return { recipe_id: _recipe for recipe_id, _recipe in recipes.items() if target_steps - window <= len(_recipe["recipe"]["instructions"]) <= target_steps + window }


def clean_step(step: str) -> str:
    cleaned_step = re.sub(r'\s*(serve\s*(immediately)?\s*and\s*)?enjoy[.!]*\s*$', '', step, flags=re.IGNORECASE)
    return cleaned_step


def clean_recipe_steps(step_obj):
    step_obj['stepText'] = clean_step(step_obj['stepText'])

    return step_obj


def filter_bad_steps(recipes):
    result = {}

    for recipe_id, recipe in recipes.items():
        new_recipe = recipe["recipe"]

        new_recipe["instructions"] = [clean_recipe_steps(step_obj) for step_obj in new_recipe["instructions"] if not bad_step(step_obj["stepText"])]

        result[recipe_id] = new_recipe 

    return recipes


def bad_step(sentence: str) -> bool:
    # at the moment only looks for sidechef in the text but more an be added in the future
    uncased_sentence = sentence.lower()
    return "sidechef" in uncased_sentence or "send to phone" in uncased_sentence or "copyright" in uncased_sentence or \
            "take a look at the video" in uncased_sentence or \
            ("enjoy" in uncased_sentence and len(uncased_sentence) <= 7) or \
            ("serve and enjoy" in uncased_sentence) or \
            ("serve immediately and enjoy" in uncased_sentence) or \
            ("well done" in uncased_sentence and len(uncased_sentence) <= 11) or \
            ("you can skip this step" in uncased_sentence) or ("photograph by" in uncased_sentence) or \
            ("photo by" in uncased_sentence) or ("find the full recipe in the video" in uncased_sentence) or \
            ("of General Mills" in uncased_sentence)


if __name__ == '__main__':
    RECIPES_FILE = 'recipes_with_image_segmented.json'
    RESULT_FILE = 'recipes_with_image_filtered_and_segmented.json'

    with open(RECIPES_FILE, "r") as f:
        data = json.load(f)

    print("Original Recipes:", len(data))

    # Filtering for size
    filtered_recipes = filter_recipe_num_steps(data)

    print("Recipes Filtered for #Steps;", len(filtered_recipes))
    print("Avg. #Steps:", average_number_steps(filtered_recipes))

    # Filtering bad steps
    filtered_recipes = filter_bad_steps(filtered_recipes)

    print("Recipes Filt. for Bad Steps;", len(filtered_recipes))
    print("Avg. #Steps:", average_number_steps(filtered_recipes))

    with open(RESULT_FILE, "w") as f:
        json.dump(filtered_recipes, f, indent=4)