import json
import os

from tqdm import tqdm

from constants import VICUNA_PROMPT_DICT, CURRENT_STEP_TEMP, NO_STEP_TEMP

CONTEXT_WINDOW = 1

SYS_TOKEN = '<|assistant|>'
USER_TOKEN = '<|prompter|>'
SEP_TOKEN = '<|endoftext|>'

dataset_name = "recipes_all_1.5"

file_name = "simulated_conversations_{split}_manual_distribution.json"

for split in ["train", "valid", "test"]:
    with open(os.path.join("data", dataset_name, file_name.format(split=split)), "r") as infile:
        dialogs = json.load(infile)

    print(f" Dialog count: {len(dialogs)}")

    processed_dialogs = []
    seen_recipes = []

    for d in dialogs:
        dialog_id = d
        d = dialogs[d]
        dialog_turns = []
        seen_recipes.append(d["task"]["recipeId"])
        recipe_context = f" ".join(
            [d["task"]["recipe"]["displayName"]] + [f"Step {s['stepNumber']}: " + s["stepText"] for s in
                                                    d["task"]["recipe"]["instructions"]])
        dialog_context = []
        current_step = ""
        for t in d["dialog"]:
            dialog_context.append({'speaker': 'user', 'text': t['user'] if t['user'] else 'start recipe.'})
            target = t['system'].replace('I looks like we have successfully', 'It looks like we have successfully')
            
            prompt = VICUNA_PROMPT_DICT.format(
                user_token=USER_TOKEN,
                sys_token=SYS_TOKEN,
                sep_token=SEP_TOKEN,
                system_tone=d['system_tone'].replace('_', ' '),
                recipe=recipe_context,
                current_step=CURRENT_STEP_TEMP.format(step_num=t['current_step']+1, step_text=d['task']['recipe']['instructions'][t['current_step']]['stepText']) if t['user'] else NO_STEP_TEMP,
                dialog=" ".join([f"{USER_TOKEN if pt['speaker'] == 'user' else SYS_TOKEN} {pt['text']} {SEP_TOKEN}" for pt in dialog_context.copy()[(-2 * CONTEXT_WINDOW) - 1:]])
            ).replace('..', '.')
            processed_dialogs.append({'dialog_id': dialog_id, 'source': prompt, 'target': target,})
            
            dialog_context.append({'speaker': 'system', 'text': t['system']})

    print(f"seen recipes: {len(seen_recipes)}")
    print(f"unq seen recipes: {len(list(set(seen_recipes)))}")

    with open(os.path.join("data", dataset_name, f"{dataset_name}_{split.replace('valid', 'eval')}.json"), "w") as outfile:
        json.dump(processed_dialogs, outfile)

    print(f"{split}: {len(processed_dialogs)}")

    print("######################")

    with open(os.path.join("data", dataset_name, f"{dataset_name}_{split.replace('valid', 'eval')}.json"), "r") as infile:
        dialogs = json.load(infile)

    prompts = []
    targets = []

    word_count = 0
    unq_words = []
    for d in tqdm(dialogs):
        if not isinstance(d['dialog_id'], str):
            print(d['dialog_id'])
        prompt = d['source']
        word_count += len(prompt.split())
        unq_words += set(prompt.split())
        prompts.append(prompt)

    print(f"words: {word_count}")
    print(f"unq words: {len(list(set(unq_words)))}")
