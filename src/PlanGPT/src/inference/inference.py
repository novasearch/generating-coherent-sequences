import json
import time
from typing import List

import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_binding import ModelArguments, InferenceArguments
from data_mod import data_utils
from inference import inference_utils
from trainers import trainer_utils


def single_example_inference(model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, example: str,
                             decoding_strategy: str, device: torch.device, max_new_tokens: int):
    inputs = torch.tensor(tokenizer.encode(example)).unsqueeze(0)
    inputs = inputs.to(device)

    if decoding_strategy == 'sampling':
        sample_outputs = model.generate(
            inputs,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            top_k=50,
            max_new_tokens=max_new_tokens,
            top_p=0.95,
            num_return_sequences=1
        )
    elif decoding_strategy == 'beam':
        sample_outputs = model.generate(
            inputs,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            num_return_sequences=1,
            early_stopping=True
        )
    else:
        sample_outputs = model.generate(
            inputs,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.pad_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            early_stopping=True
        )
    return tokenizer.decode(sample_outputs[0], skip_special_tokens=True)




def run_inference_loop(model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer,
                       dataloader: DataLoader, decoding_strategy: str, device: torch.device,
                       max_new_tokens: int) -> List[str]:
    predictions = []
    # run inference
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            batch.pop('labels', [])

            if decoding_strategy == 'sampling':
                sample_outputs = model.generate(
                    **batch,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    top_k=50,
                    max_new_tokens=max_new_tokens,
                    top_p=0.95,
                    num_return_sequences=1
                )
            elif decoding_strategy == 'beam':
                sample_outputs = model.generate(
                    **batch,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                    num_return_sequences=1,
                    early_stopping=True
                )
            else:
                sample_outputs = model.generate(
                    **batch,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.pad_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=max_new_tokens,
                    early_stopping=True
                )

            responses = tokenizer.batch_decode(sample_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            predictions.extend(responses)

            batch['input_ids'] = batch['input_ids'].to('cpu')
    return predictions


def inference():
    parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
    model_args, inference_args = parser.parse_args_into_dataclasses()  # type: (ModelArguments, InferenceArguments)
    model_args.ckpt_path = model_args.ckpt_path[:-1] if model_args.ckpt_path.endswith("/") else model_args.ckpt_path

    if inference_args.test_file is None or inference_args.test_file == "":
        print(f"ERROR: Please provide a valid test file path. Received: {inference_args.test_file}")
        return

    if model_args.ckpt_path is None or model_args.ckpt_path == "":
        print(f"ERROR: Please provide a valid ckpt file path. Received: {model_args.ckpt_path}")
        return

    print(f"Loading model and tokenizer from {model_args.ckpt_path}")

    model, tokenizer, _ = trainer_utils.load_model(model_args.base_model, model_args.ckpt_path, True, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    tokenizer.model_max_length = model.config.max_length - inference_args.max_new_tokens
    print(f"Setting tokenizer max length to {tokenizer.model_max_length}")
    model.eval()
    # Load Data
    print("Loading data...")
    test_dataset = data_utils.load_test_dataset(tokenizer, inference_args, model_args)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=4,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
    )

    start_time = time.time()
    print("Starting inference...")
    predictions = run_inference_loop(model, tokenizer, test_dataloader, inference_args.decoding_strategy, device,
                                     inference_args.max_new_tokens)

    print(f"Finished inference of {len(test_dataset)} examples in {time.time() - start_time} seconds")

    print("Moving model to cpu")
    model.to('cpu')

    output_file = model_args.ckpt_path.split('/')[-1]

    # remove the prompts from the predictions
    predictions = [prediction.strip().replace(source, '').strip() for prediction, source in zip(predictions, test_dataset.raw_sources)]

    output_dicts = []
    for idx in range(len(predictions)):
        output_dicts.append({
            'prompt': test_dataset.raw_sources[idx],
            # 'dialog_id': test_dataset.dialog_ids[idx],
            # 'system_tone': test_dataset.sys_tones[idx],
            # 'current_intent': test_dataset.intents[idx],
            'target': test_dataset.raw_targets[idx],
            'prediction': predictions[idx],
        })

    # save the targets
    with open(f"{output_file}_outputs.json", "w") as outfile:
        json.dump(output_dicts, outfile)

    if inference_args.score_metrics:
        print("Computing metrics....")

        scores = inference_utils.compute_metrics(predictions, test_dataset.raw_targets, ['bleu', 'rouge', 'meteor', 'bertscore', 'accuracy'])

        file_path = output_file + "_metrics.json"
        with open(file_path, "w+") as score_file:
            json.dump(scores, score_file, indent=4)
        print(scores)

        if inference_args.log_metrics:
            inference_utils.update_log_file(model_args.ckpt_path)


if __name__ == "__main__":
    inference()
