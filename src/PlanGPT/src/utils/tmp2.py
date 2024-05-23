# model path
import torch
import transformers

from src.inference.inference import run_inference_loop

MODEL_PATH = "experiments/PlanGPT/perpetual_vicuna_all_1.5/checkpoint_500"
# MODEL_PATH = "experiments/PlanGPT/perpetual_vicuna_v1.0/checkpoint_500"
# MODEL_PATH = "/user/data/dmgc.silva/plangpt_ckpts/perpetual_v1.0_11000"
TOKENIZER_PATH = MODEL_PATH
# TOKENIZER_PATH = "/user/data/dmgc.silva/plangpt_ckpts/perpetual_v1.0_11000"
TEST_FILE_PATH = '/user/home/dmgc.silva/plangpt_playground/data/recipes_all_1.3/recipes_all_1.3_test.json'

# load tokenizer
tokenizer = transformers.LlamaTokenizer.from_pretrained(
    TOKENIZER_PATH)

example = "<|prompter|> I will give you a recipe and I want you to help me do it step by step. Please use a very polite tone of voice. Recipe: Homemade Miso Soup with a Twist Step 1: To make the dashi, add Water to a saucepan, and soak Kombu for 30 minutes. Put saucepan on stove, and bring to a low simmer on medium heat. Once bubbles appear, remove kombu, and bring to a boil. Step 2: Meanwhile, cube the Tofu. Slice the Button Mushroom and chop the Scallion on a diagonal. Step 3: Once broth boils, turn heat down, and add Bonito Flakes. Simmer on low for 10 minutes. Strain broth through a seive into another saucepan. Add button mushroom and tofu to the broth, and bring mixture to a boil once again. Step 4: Pour 1/2 cup of the dashi into a measuring cup, and add the Red Miso Paste. Whisk until dissolved, then add back into the dashi. Garnish with some green onion. Step 5: Fry an Egg sunny side up. Then, serve atop some Japanese Rice and miso in a bowl. Garnish with green onion, Japanese Chili Powder and Sesame Seeds. Enjoy! We are on Step 3: Once broth boils, turn heat down, and add Bonito Flakes. Simmer on low for 10 minutes. Strain broth through a seive into another saucepan. Add button mushroom and tofu to the broth, and bring mixture to a boil once again. <|endoftext|> <|assistant|> ok! <|endoftext|> <|prompter|> i say next <|endoftext|> <|assistant|> Once broth boils, turn heat down, and add Bonito Flakes. Simmer on low for 10 minutes. Strain broth through a seive into another saucepan. Add button mushroom and tofu to the broth, and bring mixture to a boil once again. <|endoftext|> <|prompter|> i say next <|endoftext|> <|assistant|> "

# full precision

print("FP32: Loading model...")
# load model
model = transformers.LlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    load_in_8bit=False,
    torch_dtype=torch.float32,
)

model.to(torch.device('cuda'))
tokenizer.model_max_length = model.config.max_sequence_length - 1024

# run inference loop
outputs = run_inference_loop(model, tokenizer, [example], 'greedy', torch.device('cuda'), 1024,
                             eos_token_id=tokenizer.eos_token_id)
print(outputs[0])
