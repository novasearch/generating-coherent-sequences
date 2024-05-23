import torch
from tqdm import tqdm
import json
import os
import random
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from configs import GREEDY_CONFIG, BEAM_CONFIG, SAMPLING_CONFIG, TOP_K_CONFIG, TOP_P_CONFIG, TOP_PK_CONFIG


class InferenceModel:
    def __init__(self, peft_model_id, decoding_config=GREEDY_CONFIG):
        if not os.path.exists(os.path.join(peft_model_id, 'adapter_model.bin')):
            print("Didn't find checkpoint.")
            exit(1)

        config = PeftConfig.from_pretrained(peft_model_id)
        config.inference_mode = True

        print(config.base_model_name_or_path)

        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

        self.tokenizer = AutoTokenizer.from_pretrained(peft_model_id, fast_tokenizer=True)

        # Load the Lora model
        self.model = PeftModel.from_pretrained(base_model, peft_model_id).to("cuda")
        self.model.merge_and_unload()

        self.generation_config = decoding_config


    def construct_prompt(self, sample):
        return f"""### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n"""


    @staticmethod
    def parse_result(result):
        start = result.find("### Response:") + len("### Response:")
        end = result.find("</s>", start)

        result = result[start:end].strip()

        return result


    def infer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

        output = self.tokenizer.decode(generation_output.sequences[0])

        return output


if __name__ == '__main__':
    RECIPES_PATH = '/user/data/vfc/data/recipes/dataset/recipes_with_image_filtered.json'

    if len(sys.argv) < 4:
        print("python inference_window.py PEFT_MODEL_ID RUN_NAME CONFIG")
        exit(1)

    PEFT_MODEL_ID = sys.argv[1]
    # used to disambiguate result files
    RUN_NAME = sys.argv[2]
    CONFIG = sys.argv[3]

    STEPS_RESULT = f'/user/data/vfc/task-grounded-image-sequence-synthesis/PlanGPT/results/image_annotation/{RUN_NAME}_steps.json'
    # PROMPTS_RESULT = f'/user/data/vfc/task-grounded-image-sequence-synthesis/PlanGPT/inference_results/{config}_prompts_llm_steps.json'

    NUM_RECIPES = -1
    WINDOW = 2

    configs = {
        "greedy": GREEDY_CONFIG,
        "beam": BEAM_CONFIG,
        "sampling": SAMPLING_CONFIG,
        "top_k": TOP_K_CONFIG,
        "top_p": TOP_P_CONFIG,
        "top_pk": TOP_PK_CONFIG
    }

    model = InferenceModel(PEFT_MODEL_ID, decoding_config=configs[CONFIG])

    with open(RECIPES_PATH) as f:
        data = json.load(f)

    output = {}

    recipes = list(data.items())[:NUM_RECIPES]

    for recipe_id, _recipe in tqdm(recipes, desc="recipes"):
        recipe = _recipe['recipe']
        steps = recipe["instructions"]

        new_recipe = {"steps": [], "steps_generated": []}

        context_arr = []

        try:
            for i, step in enumerate(steps):
                if len(context_arr) < WINDOW:
                    context_arr.append(f"Step {step['stepNumber']}: {step['stepText']}\n")
                else:
                    if WINDOW == 3:
                        context_arr[0] = f"Step 1: {new_recipe['steps_generated'][i-2]}\n"
                        context_arr[1] = f"Step 2: {steps[i-1]['stepText']}\n"
                        context_arr[2] = f"Step 3: {step['stepText']}\n"
                    else:
                        assert WINDOW == 2
                        context_arr[0] = f"Step 1: {new_recipe['steps_generated'][i-1]}\n"
                        context_arr[1] = f"Step 2: {step['stepText']}\n"

                context = "".join(context_arr)

                model_input = model.construct_prompt({'instruction': "Give me the image caption for this recipe.",
                                                      'input': context})

                print("### INPUT START", model_input, "### INPUT END", sep='\n')

                result = model.infer(model_input)

                print("### RESULT START", result, "### RESULT END", sep="\n")

                result = InferenceModel.parse_result(result)

                new_recipe["steps"].append(step['stepText'])
                new_recipe["steps_generated"].append(result)
        except Exception as oom:
            print("Out of Memory Exception: ", oom)

        output[recipe_id] = new_recipe

        # dumping every recipe, cause we're looking at the result, as it generates
        json.dump(output, open(STEPS_RESULT, 'w'), indent=4)

    # prompts = []
    # for recipe_id, recipe in output.items():
    #     for i, step in enumerate(recipe['steps']):
    #         id =f"{recipe_id}_{i+1}"
    #         prompts.append({
    #             "id": id,
    #             "prompt": step
    #         })

    # with open(PROMPTS_RESULT, "w") as f:
    #     json.dump(prompts, f, indent=4)
