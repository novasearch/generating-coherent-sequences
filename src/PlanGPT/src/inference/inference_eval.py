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


    def __init__(self, peft_model_id, generation_config=GREEDY_CONFIG):
        if not os.path.exists(os.path.join(peft_model_id, 'adapter_model.bin')):
            print("Didn't find checkpoint.")
            exit(1)

        config = PeftConfig.from_pretrained(peft_model_id)
        config.inference_mode = True

        print(config.base_model_name_or_path)

        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

        self.tokenizer = AutoTokenizer.from_pretrained(peft_model_id, fast_tokenizer=True)

        # Load the LoRA model
        self.model = PeftModel.from_pretrained(base_model, peft_model_id).to("cuda")
        self.model.merge_and_unload()

        self.generation_config = generation_config


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

    if len(sys.argv) < 5:
        print("python inference_eval.py PEFT_MODEL_ID EVAL_DATASET RUN_NAME CONFIG")
        exit(1)

    PEFT_MODEL_ID = sys.argv[1]
    EVAL_DATASET =sys.argv[2]
    # used to disambiguate result files
    RUN_NAME = sys.argv[3]
    CONFIG = sys.argv[4]

    OUTPUT_FILE = f'{RUN_NAME}.json'

    NUM_SAMPLES = 200

    configs = {
        "greedy": GREEDY_CONFIG,
        "beam": BEAM_CONFIG,
        "sampling": SAMPLING_CONFIG,
        "top_k": TOP_K_CONFIG,
        "top_p": TOP_P_CONFIG,
        "top_pk": TOP_PK_CONFIG
    }

    model = InferenceModel(PEFT_MODEL_ID, generation_config=configs[CONFIG])

    with open(EVAL_DATASET) as f:
        data = json.load(f)

    output = []

    #random.seed(10)
    #random.shuffle(data)

    data = data[:NUM_SAMPLES]

    for sample in tqdm(data, desc="examples"):

        model_input = model.construct_prompt({'instruction': sample['instruction'],
                                                    'input': sample['input']})

        result = model.infer(model_input)

        result = InferenceModel.parse_result(result)

        output.append({'input': sample['input'], 'output': result})

    json.dump(output, open(OUTPUT_FILE, 'w'), indent=4)