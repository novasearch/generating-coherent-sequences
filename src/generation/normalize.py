import json
import numpy as np

from clip_similarity import text_text_similarity

PARAMETERS_FILE = 'parameters.json'

class Normalizer():
    def __init__(self):
        try:
            with open(PARAMETERS_FILE, "r") as f:
                params = json.load(f)

            self.mean = params["mean"]
            self.std = params["std"]
        except FileNotFoundError:
            print("Can't initialize parameters class")
            exit(1)

    def normalize(self, val: float) -> float:
        return (val-self.mean) / self.std


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print("Please provide a file path")
        exit(1)

    file_path = sys.argv[1]

    with open(file_path, "r") as f:
        recipes = json.load(f)

    sims = []

    for recipe_id, recipe in recipes.items():
        steps = recipe["steps_alpaca"]

        for i, step in enumerate(steps[1:]):
            sims.append(text_text_similarity(steps[i], step))

    mean = np.mean(sims)
    std = np.std(sims)

    parameters = {
        "mean": mean,
        "std": std
    }

    with open(PARAMETERS_FILE, "w") as f:
        json.dump(parameters, f, indent=4)