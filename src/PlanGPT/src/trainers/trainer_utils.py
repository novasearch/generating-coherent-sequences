import os
import shutil

import torch
import transformers
from accelerate import init_empty_weights
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import CyclicLR
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, \
    LlamaConfig, AutoConfig

from peft import PeftConfig, PeftModel

# If this fails, try this;
# https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html#save-and-load-the-model
def save_model(model: PreTrainedModel, save_dir: str, optimizer: Optimizer = None,
               tokenizer: PreTrainedTokenizer = None, state_dict=None):
    if not (os.path.exists(save_dir) and os.path.isdir(save_dir)):
        os.mkdir(save_dir)

    model.config.save_pretrained(save_dir)

    model.save_pretrained(save_dir, state_dict=state_dict)

    if not (tokenizer is None):
        tokenizer.save_pretrained(save_dir)

    if not (optimizer is None):
        optim_state_dict = {'optim_state_dict': optimizer.state_dict()}
        torch.save(optim_state_dict, save_dir + "/optim_state.pkl")


def load_model(base_model, ckpt_dir, get_tokenizer=True, get_optimizer=True, dtype=torch.float32):
    # TODO Add success or fail message idk
    optimizer = None
    tokenizer = None
    if os.path.exists(os.path.join(ckpt_dir, "config.json")) and os.path.exists(os.path.join(ckpt_dir, "model_state.pkl")):
        with init_empty_weights():
            if "alpaca" in base_model or "llama" in base_model or 'vicuna' in base_model:
                config = LlamaConfig.from_pretrained(ckpt_dir + "/config.json")
                model = LlamaForCausalLM(config)
            else:
                config = AutoConfig.from_pretrained(ckpt_dir + "/config.json")
                model = transformers.AutoModel(config)

        if os.path.isdir(ckpt_dir) and os.path.exists(ckpt_dir + "/model_state.pkl"):
            ckpt_path = ckpt_dir + "/model_state.pkl"
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            if get_optimizer and ('optim_state_dict' in checkpoint or os.path.isfile(ckpt_dir + "/optim_state.pkl")):
                print(f"Loading Optimizer from: {ckpt_path}")
                optimizer = AdamW(model.parameters())
                if 'optim_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optim_state_dict'])
                else:
                    optimizer.load_state_dict(
                        torch.load(ckpt_dir + "/optim_state.pkl", map_location=torch.device('cpu'))['optim_state_dict'])

            if get_tokenizer and os.path.isfile(ckpt_dir + "/tokenizer_config.json"):
                print(f"Loading tokenizer from: {ckpt_dir}")
                if "llama" in base_model or "alpaca" in base_model or "vicuna" in base_model:
                    tokenizer = LlamaTokenizer.from_pretrained(ckpt_dir, use_fast=False)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=False)
    else:
        if os.path.exists(os.path.join(ckpt_dir, "adapter_model.bin")):
            # Load as LoRA
            config = PeftConfig.from_pretrained(ckpt_dir)
            config.inference_mode = False
            base_model = transformers.AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

            model = PeftModel.from_pretrained(base_model, ckpt_dir)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                ckpt_dir,
                torch_dtype=dtype,
            )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            ckpt_dir,
        )

        if get_optimizer and os.path.isfile(ckpt_dir + "/optim_state.pkl"):
            print(f"Loading Optimizer from: {ckpt_dir + '/optim_state.pkl'}")
            optimizer = AdamW(model.parameters())
            optimizer.load_state_dict(
                    torch.load(ckpt_dir + "/optim_state.pkl", map_location=torch.device('cpu'))['optim_state_dict'])

    return model, tokenizer, optimizer


def _check_if_save_limit_surpassed(save_dir, save_total_limit):
    ckpt_path = "/".join(save_dir.split("/")[:-1])
    # Get the paths for all checkpoint folders
    ckpt_folders = [os.path.join(ckpt_path, d) for d in os.listdir(ckpt_path)
                    if os.path.isdir(os.path.join(ckpt_path, d)) and "checkpoint" in d]
    # Check if number of save surpasses the limit
    if "ckpts" in save_dir and len(ckpt_folders) >= save_total_limit:
        # Get date of modification for all checkpoint folders
        dir_date = [os.stat(d).st_mtime for d in ckpt_folders]
        # Get the oldest folder
        dir_to_delete = [d for d in ckpt_folders if os.stat(d).st_mtime == min(dir_date)][0]
        # Delete the folder
        print(f"Deleting folder {dir_to_delete}")
        shutil.rmtree(dir_to_delete)


def get_cyclical_schedule(optimizer: Optimizer, learning_rate: float, step_size: int = -1):
    return CyclicLR(optimizer,
                    base_lr=learning_rate,
                    max_lr=learning_rate * 5,
                    mode="triangular2",
                    gamma=0.999750031,
                    step_size_up=step_size,
                    step_size_down=step_size,
                    cycle_momentum=False)
