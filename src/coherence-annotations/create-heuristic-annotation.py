import json
import csv

STEPS_FILE = ".json"
CHOICES = ".json"

with open(STEPS_FILE, "r") as f:
    recipes = json.load(f)

with open(CHOICES, "r") as f:
    choices = json.load(f)

with open('heuristic.csv', 'w', newline='') as csvfile:
    fieldnames = ['Recipe ID', 'Steps', 'Generated Steps', 'Choices', 'Evaluation']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for recipe_id in choices:
        generated_steps = recipes[recipe_id]

        rs = []
        for i, step in enumerate(generated_steps["steps"]):
            rs.append(f"Step {i}: {step}")

        s = []
        for i, step in enumerate(generated_steps["steps_generated"]):
            s.append(f"Step {i}: {step}")

        original_steps = '\n'.join(rs)
        generated_steps = '\n'.join(s)

        choice_steps = choices[recipe_id]
        choice_list = '\n'.join(str(choice['step']) for choice in choice_steps)

        writer.writerow({
            'Recipe ID': recipe_id,
            'Steps': original_steps,
            'Generated Steps': generated_steps,
            'Choices': choice_list,
            'Evaluation': ''
        })