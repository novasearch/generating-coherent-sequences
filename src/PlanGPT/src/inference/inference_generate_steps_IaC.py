import torch
from tqdm import tqdm
import json
import os
import random
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

class InferenceModel:

    def __init__(self, peft_model_id):
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

        self.generation_config = GenerationConfig(
            temperature=0.8,
            top_p=0.75,
            top_k=40,
            num_beams=4
        )


    def construct_prompt_t5(self, sample):
        return f"""{sample['instruction']}\n{sample['input']}\n\nOutput:\n"""


    def construct_prompt(self, sample):
        return f"""### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n"""


    @staticmethod
    def parse_result(result):
        start = result.find("### Response:") + len("### Response:")
        end = result.find("</s>", start)

        return result[start:end].strip()


    def infer_t5(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(**inputs)

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


    def infer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=128
            )

        output = self.tokenizer.decode(generation_output.sequences[0])

        return output


if __name__ == '__main__':
    RECIPES_PATH = '/user/data/vfc/data/recipes/dataset/recipes_with_image.json'
    DATASET = '/user/data/vfc/visual-factual-consistency/llm-caption-finetuning/blip_instructions_and_prompt_to_caption_dataset_alpaca/blip_instructions_and_prompt_to_caption_dataset_alpaca_eval.json'
    STEPS_RESULT = '/user/data/vfc/data/recipes/dataset/steps_llm_cai.json'
    PROMPTS_RESULT = '/user/data/vfc/data/recipes/prompts/prompts_llm_steps_cai.json'

    PEFT_MODEL_ID = '/user/data/vfc/visual-factual-consistency/PlanGPT/pgpt-captions-feedback-v2-10-epochs/checkpoint_8530'

    model = InferenceModel(PEFT_MODEL_ID)

    with open(RECIPES_PATH) as f:
        data = json.load(f)

    output = {}

    # recipes = random.sample(list(data.items()), k=10)
    recipes = data.items()

    for recipe_id, _recipe in tqdm(recipes, desc="recipes"):
        new_recipe = {"steps": []}

        recipe = _recipe['recipe']

        previous_steps = ""
        try:
            for step in tqdm(recipe["instructions"], desc="steps"):

                # Current step is from the recipe
                current_step = f"Step {step['stepNumber']}: {step['stepText']}\n"

                if ("t5" in PEFT_MODEL_ID):
                    model_input = model.construct_prompt_t5({'instruction': "Give me the image caption for this recipe.", 'input': previous_steps + current_step})
                else:
                    model_input = model.construct_prompt({'instruction': "Give me the image caption for this recipe.", 'input': previous_steps + current_step})

                print("Input:")
                print(model_input)

                if "t5" in PEFT_MODEL_ID:
                    result = model.infer_t5(model_input)
                else:
                    result = model.infer(model_input)
                    result = InferenceModel.parse_result(result)

                print("Result")
                print(result)

                previous_steps += f"Step {step['stepNumber']}: {result}\n"

                new_recipe["steps"].append(result)
        except Exception as oom:
            print("Out of Memory Exception:", oom)

        output[recipe_id] = new_recipe

        # dumping every recipe, cause we're looking at the result
        json.dump(output, open(STEPS_RESULT, 'w'), indent=4)

    prompts = []
    for recipe_id, recipe in output.items():
        for i, step in enumerate(recipe['steps']):
            id =f"{recipe_id}_{i+1}"
            prompts.append({
                "id": id,
                "prompt": step
            })

    with open(PROMPTS_RESULT, "w") as f:
        json.dump(prompts, f, indent=4)
