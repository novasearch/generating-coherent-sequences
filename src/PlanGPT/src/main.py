import json
import os.path
import shutil
from typing import Dict

import torch
import torch.distributed as dist
import transformers

from constants import *
from data_mod import data_utils
from data_binding import ModelArguments, DataArguments, TrainArgs, ParallelType, PrecisionType
from trainers import trainer_utils
from trainers.fsdp_trainer import FSDPTrainer, PerpetualFSDPTrainer
from trainers.trainer import AccelTrainer, LabTrainer
from trainers.dpo_trainer import DPOTrainer
from peft import LoraConfig, TaskType, get_peft_model, LoraModel


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_model(base_model, ckpt_path=None, get_optimizer=False, model_max_length=DEFAULT_MAX_LEN, dtype=torch.float32):
    if ckpt_path is not None:
        print(f"Loading model and tokenizer from {ckpt_path}")
        model, tokenizer, _ = trainer_utils.load_model(base_model, ckpt_path, True, get_optimizer, dtype=dtype)
        model.to("cpu")
    else:
        if "alpaca" in base_model or "llama" in base_model or "vicuna" in base_model:
            print("Loading model.....")
            model = transformers.LlamaForCausalLM.from_pretrained(
                base_model,
            )

            print("Loading tokenizer.....")
            tokenizer = transformers.LlamaTokenizer.from_pretrained(
                base_model,
                model_max_length=model_max_length,
                padding_side="right",
                use_fast=False,
            )
        elif "stablelm" in base_model.lower() or "gptneox" in base_model.lower():
            print("Loading model.....")
            model = transformers.GPTNeoXForCausalLM.from_pretrained(
                base_model,
            )

            print("Loading tokenizer.....")
            tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(
                base_model,
                model_max_length=model_max_length,
                padding_side="right",
                use_fast=False,
            )
        elif "t5" in base_model.lower():
            print("Loading model.....")
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(base_model)
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                base_model,
                model_max_length=model_max_length,
                padding_side="right",
                use_fast=False)
        else:
            try:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    base_model,
                )
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    base_model,
                    model_max_length=model_max_length,
                    padding_side="right",
                    use_fast=False,
                )
            except Exception:
                raise ValueError(f"Unknown model type: {base_model}")

    tokenizer.model_max_length = model_max_length

    print(f"Tokenizer max length: {tokenizer.model_max_length}")

    print("Adding special tokens.....")
    model, tokenizer = add_special_tokens(tokenizer, model)

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have a pad token for now.")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have an eos token for now.")
    if tokenizer.bos_token_id is None:
        raise ValueError("Tokenizer must have a bos token for now.")

    return model, tokenizer


def add_special_tokens(tokenizer, model):
    tokens_dict = dict()
    if tokenizer.eos_token_id is None:
        tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token_id is None:
        tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN if tokenizer.eos_token_id is None else tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN if tokenizer.eos_token_id is None else tokenizer.eos_token
    if tokenizer.unk_token_id is None:
        tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN if tokenizer.eos_token_id is None else tokenizer.eos_token
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    return model, tokenizer


def train_w_fsdp_trainer():
    # parse arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainArgs))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()  # type: (ModelArguments, DataArguments, TrainArgs)

    print(training_args)

    # Build output_dir if not provided
    if training_args.output_dir is not None and training_args.output_dir != "":
        training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name).__str__()
    else:
        training_args.output_dir = os.path.join("/data/dmgc.silva/experiments", training_args.project_name,
                                                training_args.run_name).__str__()

    # Get current GPU and GPU_COUNT
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Check if output_dir exists, if so delete or throw warning.
    if rank == 0:
        if os.path.exists(training_args.output_dir):
            if training_args.overwrite_output_dir:
                print(f"WARNING: Directory {training_args.output_dir} already exists. OVERWRITING")
                shutil.rmtree(training_args.output_dir)
            else:
                print(f"ERROR: Directory {training_args.output_dir} already exists.")
                return

        # Create output dir
        os.makedirs(training_args.output_dir, exist_ok=True)

        # save training arguments
        json.dump(training_args.to_dict(), open(f"{training_args.output_dir}/run_arguments.json", "w"), indent=4)

    # set manual seed
    torch.manual_seed(training_args.seed)

    # Loading model and tokenizer
    model, tokenizer = load_model(model_args.base_model, ckpt_path=model_args.ckpt_path,
                                  get_optimizer=training_args.reload_optimizer,
                                  model_max_length=training_args.model_max_length,
                                  dtype=torch.float16 if training_args.mixed_precision in [PrecisionType.FP16, PrecisionType.BF16] else torch.float32)
    # peft
    if training_args.use_dpo or training_args.lora:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32,
                                 lora_dropout=0.1)
        model = get_peft_model(model, peft_config)

    # Load dataset
    train_dataloader, eval_dataloader = data_utils.load_train_data(tokenizer, data_args, training_args,
                                                                   init_process_group=world_size > 1,
                                                                   model_name=model_args.base_model if model_args.ckpt_path is None else model_args.ckpt_path)

    if training_args.parallel_type == ParallelType.DP:
        if training_args.perpetual:
            raise NotImplementedError("Perpetual DP is not supported yet.")
        else:
            trainer = AccelTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                optimizer=None,
                args=training_args
            )
    elif training_args.parallel_type == ParallelType.FSDP:
        if training_args.perpetual:
            trainer = PerpetualFSDPTrainer(
                model=model,
                local_rank=local_rank,
                rank=rank,
                world_size=world_size,
                tokenizer=tokenizer,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                optimizer=None,
                args=training_args,
                data_args=data_args
            )
        else:
            trainer = FSDPTrainer(
                model=model,
                local_rank=local_rank,
                rank=rank,
                world_size=world_size,
                tokenizer=tokenizer,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                optimizer=None,
                args=training_args,
                data_args=data_args
            )
    else:
        if training_args.perpetual:
            raise NotImplementedError("Perpetual training is not supported yet, only for FDSP.")
        elif training_args.use_dpo:

            trainer = DPOTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                optimizer=None,
                args=training_args,
                data_args=data_args
            )

        else:
            trainer = LabTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                optimizer=None,
                args=training_args,
                data_args=data_args
            )

    trainer.train()
    cleanup()


if __name__ == "__main__":
    train_w_fsdp_trainer()
