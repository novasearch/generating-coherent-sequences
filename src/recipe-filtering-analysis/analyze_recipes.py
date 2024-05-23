import json
from tqdm import tqdm
import pprint as pp
import numpy as np
import sys


def average_step_length(recipes, words=False):
        def num_words(string):
            return len(string.split(" "))

        steps = [step["stepText"] for _, _recipe in recipes.items() for step in _recipe["recipe"]["instructions"]]

        total_avg = sum(map(num_words if words else len, steps)) / len(steps)

        return total_avg


def average_number_steps(recipes):
    steps = [len(_recipe["recipe"]["instructions"]) for _, _recipe in recipes.items()]

    total_avg = sum(steps) / len(steps)

    return total_avg


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage: python analyze_recipes.py <path_0>, ..., <path_n>")
        exit(1)

    for path in sys.argv[1:]:
        with open(path, "r") as f:
            data = json.load(f)

        print(f"File: {path.split('/')[-1]}")
        print(f"#Recipes: {len(data)}")
        print(f"Avg. #Steps: {average_number_steps(data):.2f}")
        print(f"Avg. #Chars: {average_step_length(data):.2f}")
        print(f"Avg. #Words: {average_step_length(data, True):.2f}")
        print("-" * 10)
