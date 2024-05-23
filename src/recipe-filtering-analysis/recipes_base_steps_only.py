import json
from tqdm import tqdm

RECIPES_FILE = "recipes_base.json"
RECIPES_RESULT = "recipes_base_steps_only.json"

n = "\n"

with open(RECIPES_FILE) as f:
    data = json.load(f)

output = {}

for recipe in tqdm(list(data.values())):
    new_recipe = {"steps": [ 
        step["step_text"] for step in recipe["instructions"]
    ]}

    output[recipe["doc_id"]] = new_recipe

with open(RECIPES_RESULT, "w") as f:
    json.dump(output, f, indent=4)