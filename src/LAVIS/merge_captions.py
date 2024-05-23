import json

RESULT = ".json"

path = lambda caption_type: f"captions_{caption_type}.json"

PROMPT_TYPES = ["no_context", "current_step", "full_context", "last_step", "window"]

paths = [path(caption_type) for caption_type in PROMPT_TYPES]

dicts = []

for p in paths:
    try:
        with open(p, "r") as f:
            dicts.append(json.load(f))
    except FileNotFoundError:
        continue

final_dict = {}

for d in dicts:
    for key, value in d.items():
        final_dict[key] = final_dict.get(key, {})
        final_dict[key].update(value)

with open(RESULT, "w") as f:
    json.dump(final_dict, f, indent=4)