import json
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Optional, Union

from transformers.utils import ExplicitEnum

from constants import *
from data_mod.datasets import DatasetType


class CustomIntervalStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"
    BEST_EVAL = "best_eval"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        if isinstance(other, CustomIntervalStrategy):
            return self.value == other.value
        return False


class RunTypes(ExplicitEnum):
    pretrain = 'pretrain'
    train = 'train'
    inference = 'inference'

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return False


class DecodingStrategies(ExplicitEnum):
    greedy = 'greedy'
    beam = 'beam'
    sampling = 'sampling'

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return False


class PrecisionType(ExplicitEnum):
    BF16 = 'BF16'
    FP16 = 'FP16'
    FP32 = 'FP32'

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value.lower() == other.lower()
        if isinstance(other, PrecisionType):
            return self.value.lower() == other.value.lower()
        return False


class CustomSchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WARM_RESTARTS = "cosine_warm_restarts"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    CYCLICAL = "cyclical"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other.lower()
        if isinstance(other, CustomSchedulerType):
            return self.value == other.value
        return False


class ParallelType(ExplicitEnum):
    NO = "no"
    DP = "dp"
    FSDP = "fsdp"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other.lower()
        if isinstance(other, CustomSchedulerType):
            return self.value == other.value
        return False


@dataclass
class ModelArguments:
    base_model: str = field(
        metadata={"help": "The name of the model used."},
    )
    ckpt_path: str = field(
        default=None,
        metadata={"help": "The path of the model to be loaded."},
    )


@dataclass
class DataArguments:
    data_path: str = field(default="./data", metadata={"help": "Path to the training data."})
    dataset_name: str = field(default=None, metadata={"help": "The name of the dataset folder"})
    dataset_type: DatasetType = field(default=None, metadata={"help": "The type of the dataset"})


@dataclass
class TrainArgs:
    experiment_name: str = field(
        default="debug",
        metadata={"help": "The name of this specific run to be used in WANDB."}
    )
    reload_optimizer: bool = field(
        default=False,
        metadata={"help": "If the optimizer should be reloaded from checkpoint."},
    )
    save_optimizer: bool = field(
        default=False,
        metadata={"help": "If the optimizer should be saved on checkpoint."},
    )
    cache_dir: Optional[str] = field(
        default=None
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Run name to be used in wandb"}
    )
    optim: str = field(
        default="adamw_torch"
    )
    model_max_length: int = field(
        default=DEFAULT_MAX_LEN,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    output_dir: str = field(
        default='',
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    project_name: str = field(
        default='debug',
        metadata={"help": "The name of the project in WANDB."},
    )
    num_train_epochs: float = field(
        default=DEFAULT_EPOCHS,
        metadata={"help": "The number of epochs to train for."},
    )
    per_device_train_batch_size: int = field(
        default=DEFAULT_BATCH_SIZE,
        metadata={"help": "The batch size used for each GPU during training."},
    )
    per_device_eval_batch_size: int = field(
        default=DEFAULT_BATCH_SIZE,
        metadata={"help": "The batch size used for each GPU during evaluation."},
    )
    save_strategy: Union[CustomIntervalStrategy, str] = field(
        default=CustomIntervalStrategy.EPOCH.value,
        metadata={"help": "The checkpoint save strategy to use."},
    )
    evaluation_strategy: Union[CustomIntervalStrategy, str] = field(
        default=CustomIntervalStrategy.EPOCH.value,
        metadata={"help": "The evaluation strategy to use."},
    )
    run_type: Union[RunTypes, str] = field(
        default='train',
        metadata={"help": "Whether we are running train, inference or pretrain"},
    )
    report_to_wandb: bool = field(
        default=False,
        metadata={"help": "If the logs should be reported to WANDB"},
    )
    seed: int = field(
        default=DEFAULT_SEED,
        metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    stop_criteria: float = field(
        default=DEFAULT_STOP_CRITERIA,
        metadata={"help": "The eval loss lower bound."},
    )
    lr_scheduler_type: Union[CustomSchedulerType, str] = field(
        default="cosine",
        metadata={"help": "The scheduler type to use."},
    )
    lr_step_size: int = field(
        default=DEFAULT_LR_STEP_SIZE,
        metadata={"help": "For cyclical lr, 2 / the number of steps between peaks."},
    )
    perpetual: bool = field(
        default=False,
        metadata={"help": "If set then the training will never stop. Only when killed."},
    )
    debug: bool = field(
        default=False,
        metadata={"help": "Debug."},
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    infer_checkpoints: bool = field(
        default=False,
        metadata={"help": "If set then the inference will be performed on all the checkpoints in the output dir."},
    )
    learning_rate: float = field(default=DEFAULT_LEARNING_RATE,
                                 metadata={"help": "The initial learning rate for AdamW."})
    mixed_precision: Union[PrecisionType, str] = field(
        default=PrecisionType.FP32.value,
        metadata={"help": "The mixed precision type to use."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    eval_steps: Optional[int] = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    save_steps: Optional[int] = field(default=None, metadata={"help": "Save the model every X steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_before_inference: int = field(
        default=0, metadata={"help": "Number of steps before queueing the first inference."}
    )
    param_update_strategy: Union[CustomIntervalStrategy, str] = field(
        default=CustomIntervalStrategy.EPOCH.value,
        metadata={"help": "The hyperparameter update strategy to use."},
    )
    param_update_interval: int = field(
        default=500, metadata={"help": "Number of steps between hyperparameter updates."}
    )
    parallel_type: str = field(
        default=ParallelType.NO.value, metadata={"help": "The type of parallelism to use (NO, DP, FDSP)."}
    )
    use_dpo: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use Direct Preference Optimization (DPO)"
            )
        },
    )
    dpo_mixed: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not add ce_loss to DPO loss"
            )
        },
    )
    dpo_beta: float = field(
        default=0.1, metadata={"help": "The Beta value for DPO Trainer"}
    )
    dpo_theta: float = field(
        default=0.2, metadata={"help": "The Theta value for DPO Trainer"}
    )
    interval_dpo: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to interval DPO"
            )
        },
    )
    interval_dpo_steps: int = field(
        default=4, metadata={"help": "The number of steps between DPO steps. Eg if 4 then DPO happens every 4 steps"}
    )
    lora: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to load model with lora"
            )
        },
    )


    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)


@dataclass
class InferenceArguments:
    test_file: str = field(
        default=None,
        metadata={"help": "The path to the test file to be used during inference."},
    )
    decoding_strategy: Union[DecodingStrategies, str] = field(
        default=DecodingStrategies.beam.value,
        metadata={"help": "The decoding strategy to be used during inference."},
    )
    score_metrics: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute BLUE, METEOR, and ROUGE after inference."},
    )
    max_new_tokens: int = field(
        default=DEFAULT_MAX_NEW_TOKENS,
        metadata={"help": "The maximum number of new tokens to be generated during inference."},
    )
    max_examples: int = field(
        default=-1,
        metadata={"help": "The maximum number of examples to be considered during inference."},
    )
    log_metrics: bool = field(
        default=False,
        metadata={"help": "If score_metrics is set, it will store the metrics results in a csv on the parent folder."},
    )
    dataset_type: DatasetType = field(default=None, metadata={"help": "The type of the dataset"})
