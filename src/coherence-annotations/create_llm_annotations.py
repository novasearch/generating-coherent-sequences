import json
import random
import csv

if __name__ == '__main__':
    LLM_RESULTS = ".json"

    ANNOTATORS = 6
    ANNOTATIONS_PER_PERSON = 30
    REPEATS = 3

    with open(LLM_RESULTS, "r") as f:
        data = json.load(f)

    # Create a map based on the number of steps in the input
    step_map = {}

    for i, item in enumerate(data):
        steps = item["input"].count("Step")
        if steps not in step_map:
            step_map[steps] = []
        item["id"] = i
        step_map[steps].append(item)

    step_map = {key: value for key, value in step_map.items() if key in [1, 2, 3]}

    for k in step_map:
        print(f"{k}: {len(step_map[k])}")

    data = []
    for v in step_map.values():
        data.extend(v)

    random.seed(10)
    random.shuffle(data)

    print(f"{len(data)} examples.")

    S = ANNOTATORS // REPEATS

    subsets = []

    # Create S contiguous subsets of size A
    for i in range(S):
        subset = data[i * ANNOTATIONS_PER_PERSON : (i + 1) * ANNOTATIONS_PER_PERSON]
        subsets.append(subset)

    assignments = []

    for i in range(ANNOTATORS):
        assignments.append(subsets[i % S])

    print(len(assignments))

    for assignment in assignments:
        print(len(assignment))

    for assignment in assignments:
        ids = [a['id'] for a in assignment]
        print(ids)

    # Create a CSV file
    with open(f'llm_annotations/c0c.csv', 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Input', 'Prompt', 'Rating']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        for assignment in assignments:
            for item in assignment:
                writer.writerow({
                    'Id': item['id'],
                    'Input': item['input'],
                    'Prompt': item['output'],
                    'Rating': ''
                })