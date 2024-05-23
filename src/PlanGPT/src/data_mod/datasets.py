import copy
import json
import os
from enum import Enum
from typing import Dict, Sequence, List

import numpy as np
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from data_mod.data_constants import CONVERSATIONAL_PROMPT_FORMAT
from .data_constants import IGNORE_INDEX


class DatasetType(str, Enum):
    DIALOGS_DATASET = "dialogs_dataset"
    DPO_DIALOGS_DATASET = "dpo_dialogs_dataset"
    OPTDataset = "opt_dataset"
    CAPTIONS_DATASET = "captions_dataset"
    CAPTIONS_DATASET_T5 = "captions_dataset_t5"

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, DatasetType):
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other
        return False


def print_data_metrics(lens):
    if int(os.environ['RANK']) == 0:
        print(f"-------------> Data Metrics: <-------------")
        print(f"Max length: {max(lens)}")
        print(f"Min length: {min(lens)}")
        print(f"Mean length: {sum(lens) / len(lens)}")
        print(f"90th percentile: {int(np.percentile(lens, 90))}")
        print(f"95th percentile: {int(np.percentile(lens, 95))}")
        print(f"99th percentile: {int(np.percentile(lens, 99))}")


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids.ne(tokenizer.pad_token_id).sum().item()
        for text in strings
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    print_data_metrics(sources_tokenized["input_ids_lens"])
    return dict(input_ids=input_ids, labels=labels)


def preprocess_seq2seq(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict:
        """Preprocess the data by tokenizing."""
        # examples = [s + t for s, t in zip(sources, targets)]
        examples = targets
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        input_ids = sources_tokenized["input_ids"]
        labels = examples_tokenized["input_ids"]
        print(tokenizer.decode(input_ids[0]))
        print(tokenizer.decode(labels[0]))

        # labels = copy.deepcopy(input_ids)
        # for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        #     label[:source_len] = IGNORE_INDEX
        print_data_metrics(sources_tokenized["input_ids_lens"])
        print_data_metrics(examples_tokenized["input_ids_lens"])
        
        return dict(input_ids=input_ids, labels=labels)


class DialogsDataset(Dataset):

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, debug=False, **kwargs):
        # load files
        with open(data_path, "r") as infile:
            dialogs = json.load(infile)

        prompts = []
        targets = []

        if debug:
            dialogs = dialogs[:1024]

        model_name = kwargs.get('model_name', 'llama')

        if 'vicuna' in model_name or 'oa' in model_name:
            for d in dialogs:
                p = CONVERSATIONAL_PROMPT_FORMAT.format(
                    recipe=d["recipe"],
                    current_step=d['current_step'],
                    system_tone=d['system_tone'].replace('_', ' '),
                    dialog=' '.join(d["dialog"]).replace("System:", " <|endoftext|> <|assistant|>").replace("User:", "<|prompter|>")
                )
                targets.append(f"{d['response'].replace('User:', '<|prompter|>')} <|endoftext|> {tokenizer.eos_token}")
                prompts.append(p)
        else:
            for d in dialogs:
                prompt = d["prompt"].format(recipe=d["recipe"], current_step=d['current_step'],
                                            dialog=' '.join(d["dialog"]))
                prompts.append(prompt)
                targets.append(f" {d['response']}{tokenizer.eos_token}")

        # tokenize
        data_dict = preprocess(prompts, targets, tokenizer)

        self.raw_sources = prompts
        self.raw_targets = targets

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.dialog_ids = [d["dialog_id"] for d in dialogs]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], dialog_ids=self.dialog_ids[i])


class CaptionsDataset(Dataset):

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, debug=False, **kwargs):
        # load files
        with open(data_path, "r") as infile:
            dataset = json.load(infile)

        prompts = []
        targets = []

        if debug:
            dataset = dataset[:100]

        for d in dataset:
            prompt = f"""### Instruction:\n{d['instruction']}\n\n### Input:\n{d['input']}\n\n### Response:\n"""
            prompts.append(prompt)
            targets.append(f"{d['output']}{tokenizer.eos_token}")

        # tokenize
        data_dict = preprocess(prompts, targets, tokenizer)

        self.raw_sources = prompts
        self.raw_targets = targets

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class CaptionsDatasetT5(Dataset):

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, debug=False, **kwargs):
        # load files
        with open(data_path, "r") as infile:
            dataset = json.load(infile)

        prompts = []
        targets = []

        if debug:
            dataset = dataset[:100]

        for d in dataset:
            prompt = f"{d['instruction']}:\n{d['input']}"
            prompts.append(prompt)
            targets.append(f"{d['output']}{tokenizer.eos_token}")

        # tokenize
        data_dict = preprocess_seq2seq(prompts, targets, tokenizer)

        self.raw_sources = prompts
        self.raw_targets = targets

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def dpo_preprocess(sources: List[str], targets: List[str], negative_targets: List[str], tokenizer: PreTrainedTokenizer):
    # tokenize sources
    tokenized_sources = [tokenizer(
        source,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=False,
    ).input_ids[0] for source in sources]

    # tokenize targets
    tokenized_targets = [tokenizer(
        target,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=False,
    ).input_ids[0] for target in targets]

    # tokenize negative targets
    tokenized_negative_targets = [tokenizer(
        target,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=False,
    ).input_ids[0] for target in negative_targets]

    # concatenate sources and targets
    tokenized_sources_and_targets = [torch.cat((source, target)) for source, target in zip(tokenized_sources, tokenized_targets)]
    labels = copy.deepcopy(tokenized_sources_and_targets)

    # mask sources
    for t, s in zip(labels, tokenized_sources):
        t[:len(s)] = IGNORE_INDEX

    # concatenate sources and negative targets
    tokenized_sources_and_negative_targets = [torch.cat((source, target)) for source, target in zip(tokenized_sources, tokenized_negative_targets)]
    negative_labels = copy.deepcopy(tokenized_sources_and_negative_targets)

    # mask sources
    for t, s in zip(negative_labels, tokenized_sources):
        t[:len(s)] = IGNORE_INDEX

    # prune to max length (right side truncation)
    tokenized_sources_and_targets = [t[-tokenizer.model_max_length:] for t in tokenized_sources_and_targets]
    tokenized_sources_and_negative_targets = [t[-tokenizer.model_max_length:] for t in tokenized_sources_and_negative_targets]
    labels = [t[-tokenizer.model_max_length:] for t in labels]
    negative_labels = [t[-tokenizer.model_max_length:] for t in negative_labels]

    print_data_metrics([len(s) for s in tokenized_sources_and_targets])

    # pad
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokenized_sources_and_targets = pad_sequence(tokenized_sources_and_targets, batch_first=True, padding_value=pad_token_id)
    tokenized_sources_and_negative_targets = pad_sequence(tokenized_sources_and_negative_targets, batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=pad_token_id)
    negative_labels = pad_sequence(negative_labels, batch_first=True, padding_value=pad_token_id)

    return dict(
        input_ids_and_labels=tokenized_sources_and_targets,
        input_ids_and_negative_labels=tokenized_sources_and_negative_targets,
        labels=labels,
        negative_labels=negative_labels
    )


class DPODialogsDataset(Dataset):

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, debug=False, **kwargs):
        # load files
        with open(data_path, "r") as infile:
            dialogs = json.load(infile)

        prompts = []
        targets = []
        negative_targets = []

        if debug:
            dialogs = dialogs[:4096]

        for d in dialogs:
            p = CONVERSATIONAL_PROMPT_FORMAT.format(
                recipe=d["recipe"],
                current_step=d['current_step'],
                system_tone=d['system_tone'].replace('_', ' '),
                dialog=' '.join(d["dialog"]).replace("System:", " <|fakeeos|> <|assistant|>").replace("User:", "<|prompter|>")
            ).replace("<|endoftext|>", "<|fakeeos|>")
            targets.append(f"{d['response'].replace('User:', '<|prompter|>')} <|fakeeos|> {tokenizer.eos_token}")
            negative_targets.append(f"{d['negative_response'].replace('User:', '<|prompter|>')} <|fakeeos|> {tokenizer.eos_token}")
            prompts.append(p)

        # tokenize
        data_dict = dpo_preprocess(prompts, targets, negative_targets, tokenizer)

        self.raw_sources = prompts
        self.raw_targets = targets
        self.raw_negative_targets = negative_targets

        self.input_ids_and_labels = data_dict["input_ids_and_labels"]
        self.input_ids_and_negative_labels = data_dict["input_ids_and_negative_labels"]
        self.labels = data_dict["labels"]
        self.labels_mask = [(l != IGNORE_INDEX) & (l != tokenizer.pad_token_id) for l in self.labels]

        self.negative_labels = data_dict['negative_labels']
        self.negative_labels_mask = [(l != IGNORE_INDEX) & (l != tokenizer.pad_token_id) for l in self.negative_labels]

        # print(self.raw_sources[0])
        # print(tokenizer.decode(self.input_ids_and_labels[0], skip_special_tokens=True))
        # print(" ====================== ")
        # print(self.raw_targets[0])
        # self.labels[0][~self.labels_mask[0]] = tokenizer.pad_token_id
        # print(tokenizer.decode(self.labels[0], skip_special_tokens=True))
        # print(" ====================== ")
        # self.negative_labels[0][~self.negative_labels_mask[0]] = tokenizer.pad_token_id
        # print(self.raw_negative_targets[0])
        # print(tokenizer.decode(self.negative_labels[0], skip_special_tokens=True))
        # print(" ====================== ")
        # print(tokenizer.decode(self.input_ids_and_negative_labels[0], skip_special_tokens=True))

    def __len__(self):
        return len(self.input_ids_and_labels)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids_and_labels[i], labels=self.labels[i], negative_labels=self.negative_labels[i],
                    labels_mask=self.labels_mask[i], negative_labels_mask=self.negative_labels_mask[i],
                    input_ids_and_negative_labels=self.input_ids_and_negative_labels[i])

class OPTDataset(Dataset):

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, debug=False, **kwargs):
        # load files
        with open(data_path, "r") as infile:
            dialogs = json.load(infile)

        prompts = []
        targets = []
        negative_targets = []

        if debug:
            dialogs = dialogs[:4096]
        else:
            dialogs = dialogs[:int(len(dialogs)*0.2)]

        for d in dialogs:
            prompts.append(d["prompt"])
            targets.append(d["chosen"] + f" {tokenizer.eos_token}")
            if d["rejected"] is None:
                negative_targets.append(f"Sorry I am not sure about that. {tokenizer.eos_token}")
            else:
                negative_targets.append(d["rejected"] + f" {tokenizer.eos_token}")

        # tokenize
        data_dict = dpo_preprocess(prompts, targets, negative_targets, tokenizer)

        self.raw_sources = prompts
        self.raw_targets = targets
        self.raw_negative_targets = negative_targets

        self.input_ids_and_labels = data_dict["input_ids_and_labels"]
        self.input_ids_and_negative_labels = data_dict["input_ids_and_negative_labels"]
        self.labels = data_dict["labels"]
        self.labels_mask = [(l != IGNORE_INDEX) & (l != tokenizer.pad_token_id) for l in self.labels]

        self.negative_labels = data_dict['negative_labels']
        self.negative_labels_mask = [(l != IGNORE_INDEX) & (l != tokenizer.pad_token_id) for l in self.negative_labels]

        # print(self.raw_sources[0])
        # print(tokenizer.decode(self.input_ids_and_labels[0], skip_special_tokens=True))
        # print(" ====================== ")
        # print(self.raw_targets[0])
        # self.labels[0][~self.labels_mask[0]] = tokenizer.pad_token_id
        # print(tokenizer.decode(self.labels[0], skip_special_tokens=True))
        # print(" ====================== ")
        # self.negative_labels[0][~self.negative_labels_mask[0]] = tokenizer.pad_token_id
        # print(self.raw_negative_targets[0])
        # print(tokenizer.decode(self.negative_labels[0], skip_special_tokens=True))
        # print(" ====================== ")
        # print(tokenizer.decode(self.input_ids_and_negative_labels[0], skip_special_tokens=True))

    def __len__(self):
        return len(self.input_ids_and_labels)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids_and_labels[i], labels=self.labels[i], negative_labels=self.negative_labels[i],
                    labels_mask=self.labels_mask[i], negative_labels_mask=self.negative_labels_mask[i],
                    input_ids_and_negative_labels=self.input_ids_and_negative_labels[i])


TYPE_TO_DATASET_CLASS = {
    DatasetType.DIALOGS_DATASET.value: DialogsDataset,
    DatasetType.DPO_DIALOGS_DATASET.value: DPODialogsDataset,
    DatasetType.OPTDataset.value: OPTDataset,
    DatasetType.CAPTIONS_DATASET.value: CaptionsDataset,
    DatasetType.CAPTIONS_DATASET_T5.value: CaptionsDatasetT5
}
