import json

FILE = ".json"

with open(FILE, "r") as f:
    data = json.load(f)

final_result = {}

for key, value in data.items():
    # Extract the ID from the key (e.g., "0" from "0_1")
    step_id = key.split('_')[0]

    # If the ID doesn't exist in the final result, create a dictionary for it
    if step_id not in final_result:
        final_result[step_id] = {"steps": [], "window": []}

    # Append the step description and window to the corresponding arrays
    final_result[step_id]["steps"].append(value["step_description"])
    final_result[step_id]["window"].append(value["window"])


with open("temp.json", "w") as f:
    json.dump(final_result, f, indent=4)