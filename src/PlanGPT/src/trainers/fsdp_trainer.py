import functools
import json
import math
import os.path
import shutil
import subprocess
import time
from distutils.version import LooseVersion


import torch
import torch.distributed as dist
import torch.optim as optim
import tqdm
import transformers
import wandb

from data_mod import data_utils
from data_binding import DataArguments, TrainArgs, CustomIntervalStrategy, \
    CustomSchedulerType, PrecisionType
from torch.cuda import nccl
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import trainers.trainer_utils as trainer_utils

fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float16,
    # Buffer precision.
    buffer_dtype=torch.float16,
)

bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)

fp32_policy = MixedPrecision(
    param_dtype=torch.float32,
    # Gradient communication precision.
    reduce_dtype=torch.float32,
    # Buffer precision.
    buffer_dtype=torch.float32,
)


class FSDPTrainer:
    def __init__(self,
                 model: PreTrainedModel,
                 local_rank: int,
                 rank: int,
                 world_size: int,
                 tokenizer: PreTrainedTokenizer,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 optimizer: Optimizer = None,
                 args: TrainArgs = None,
                 data_args: DataArguments = None):
        self.model = model
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.args = args
        self.optimizer = optimizer
        self.data_args = data_args

        # Set wrap policy
        t5_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                LlamaDecoderLayer,
            },
        )
        # set sharding strat
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD  # for Zero2 and FULL_SHARD for Zero3
        torch.cuda.set_device(local_rank)

        # Check if BF16 is supported
        bf16_ready = (
                torch.version.cuda
                and torch.cuda.is_bf16_supported()
                and LooseVersion(torch.version.cuda) >= "11.0"
                and dist.is_nccl_available()
                and nccl.version() >= (2, 10)
        )

        # Set Mixed Precision type
        # TODO use training_args.bf16, training_args.fp16, training_args.fp32
        if args.mixed_precision == PrecisionType.BF16 and bf16_ready:
            mp_policy = bfSixteen
        elif args.mixed_precision == PrecisionType.FP16:
            mp_policy = fpSixteen
        else:
            mp_policy = None  # defaults to fp32

        # Distribute the model
        # model is on CPU before input to FSDP
        self.model = FSDP(model,
                          auto_wrap_policy=t5_auto_wrap_policy,
                          mixed_precision=mp_policy,
                          sharding_strategy=sharding_strategy,
                          device_id=torch.cuda.current_device())

        # Build Optimizer
        if self.optimizer is None:
            self.optimizer = optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon)

        # Compute Stats
        self.batches_per_epoch = (len(train_dataloader) // args.gradient_accumulation_steps)
        self.batches_per_epoch = max(self.batches_per_epoch, 1)

        if args.max_steps > 0:
            self.total_steps = args.max_steps
            self.num_train_epochs = args.max_steps // self.batches_per_epoch + int(
                args.max_steps % self.batches_per_epoch > 0
            )
        else:
            self.total_steps = math.ceil(args.num_train_epochs * self.batches_per_epoch)
            self.num_train_epochs = math.ceil(args.num_train_epochs)

        self.eval_batches = len(eval_dataloader) if eval_dataloader else 0

        self.warmup_steps = args.warmup_steps if args.warmup_steps >= 0 else int(
            self.batches_per_epoch * self.args.warmup_ratio)

        # Build Scheduler
        if args.lr_scheduler_type == CustomSchedulerType.CYCLICAL or args.lr_scheduler_type == "cyclical":
            self.scheduler = trainer_utils.get_cyclical_schedule(self.optimizer, args.learning_rate,
                                                                 args.lr_step_size)
        else:
            self.scheduler = transformers.get_scheduler(args.lr_scheduler_type, self.optimizer,
                                                        self.warmup_steps, self.total_steps)
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.saved_checkpoints = []

        if args.report_to_wandb:
            self._init_wandb()

    def _init_wandb(self):
        if self.rank == 0:
            wandb.init(project=self.args.project_name, config=self.args.to_dict())
            wandb.run.name = self.args.run_name

    def train(self):
        self.print(" +++++++++++++++++++++++ RUN STATISTICS +++++++++++++++++++++++")
        self.print(f"Epochs: {self.num_train_epochs}")
        self.print(f"Batch Size: {self.args.per_device_train_batch_size}")
        self.print(f"Batches per Epoch: {self.batches_per_epoch}")
        self.print(f"Evaluation Batches: {self.eval_batches}")
        self.print(f"Warmup Steps: {self.warmup_steps}")
        self.print(f"Total Train Steps: {self.total_steps}")
        self.print(" ++++++++++++++++++++++++++ TRAINING ++++++++++++++++++++++++++")

        for epoch in range(1, int(self.num_train_epochs) + 1):
            t0 = time.time()
            self.print(f"Epoch {epoch} starting...")
            self.print("--> entering train loop")
            train_loss = self.train_loop(epoch)

            self.print(f"--> train done")

            if self.args.evaluation_strategy == CustomIntervalStrategy.EPOCH:
                self.print(f"--> running validation for epoch {epoch}")
                eval_loss = self.validation_loop()
                if self.args.save_strategy == CustomIntervalStrategy.BEST_EVAL and self.best_val_loss > eval_loss:
                    self.best_val_loss = eval_loss
                    self.print(f"--> saving best model with loss: {self.best_val_loss}")
                    self.save_checkpoint()
            if self.args.save_strategy == CustomIntervalStrategy.EPOCH:
                self.print(f"--> saving on epoch {epoch}")
                ckpt_name = self.save_checkpoint()
                if self.args.infer_checkpoints and self.rank == 0:
                    subprocess.call(['sbatch', 'scripts/infer_checkpoint.sh', os.path.join(self.args.output_dir, ckpt_name),
                                     os.path.join(self.data_args.data_path, self.data_args.dataset_name, self.data_args.dataset_name + "_eval.json")])
            self.print(f"--> epoch {epoch} completed")

        dist.barrier()

    def train_loop(self, epoch, callback=None):
        self.model.train()
        fsdp_loss = torch.zeros(2).to(self.local_rank)

        inner_pbar = tqdm.tqdm(
            range(self.batches_per_epoch), colour="blue", desc=f"Training Epoch {epoch}", disable=self.rank != 0
        )

        if self.rank == 0:
            start_step = self.global_step
            start_time = time.time()
        bad_batches = []
        for step, batch in enumerate(self.train_dataloader):
            with torch.set_grad_enabled(True):
                _ = batch.pop("dialog_ids", [])
                loss = self.training_step(batch)

                loss = self.cleanup_loss(loss, fsdp_loss)

                fsdp_loss[0] += loss
                fsdp_loss[1] += 1

                if (step + 1) % self.args.gradient_accumulation_steps == 0 \
                        or step == self.batches_per_epoch - 1:
                    self.global_step += 1
                    self.gradient_accumulation()
                    inner_pbar.update(1)
                    if self.args.report_to_wandb and self.global_step % self.args.logging_steps == 0 and self.rank == 0:
                        self._log(start_time, fsdp_loss[0].item(), self.global_step - start_step, phase="train")
                    if self.args.evaluation_strategy == CustomIntervalStrategy.STEPS and self.global_step % self.args.eval_steps == 0:
                        self.print(f"--> running validation@{self.global_step}")
                        _ = self.validation_loop()
                        self.model.train()
                    if self.args.save_strategy == CustomIntervalStrategy.STEPS and self.global_step % self.args.save_steps == 0:
                        self.print(f"--> saving model@{self.global_step}")
                        ckpt_name = self.save_checkpoint()
                        if self.args.infer_checkpoints and self.rank == 0 and self.global_step >= self.args.warmup_before_inference:
                            subprocess.call(['sbatch', 'scripts/infer_checkpoint.sh',
                                             os.path.join(self.args.output_dir, ckpt_name),
                                             os.path.join(self.data_args.data_path, self.data_args.dataset_name, self.data_args.dataset_name + "_eval.json")])

                if callback is not None and self.global_step % self.args.param_update_interval == 0 and self.args.param_update_strategy == CustomIntervalStrategy.STEPS:
                    self.print(f"--> updating params@{self.global_step}")
                    callback(self)

                if self.global_step == self.total_steps:
                    self.print(f"--> stopping training@{self.global_step}")
                    break
        print(f"before reduce: fsdp_loss: {fsdp_loss} avg: {fsdp_loss[0].item() / fsdp_loss[1].item()} rank: {self.rank} local_rank: {self.local_rank} device: {torch.cuda.current_device()}")
        dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
        self.print(f"After reduce: fsdp_loss: {fsdp_loss} avg: {fsdp_loss[0].item() / fsdp_loss[1].item()}")
        train_loss = fsdp_loss[0].item() / fsdp_loss[1].item()

        inner_pbar.close()
        self.print(
            f"--> train epoch: \t{epoch}, loss: \t{train_loss:.4f}"
        )
        # report metrics to wandb
        if self.args.report_to_wandb and self.rank == 0:
            self._log(start_time, fsdp_loss[0].item(), fsdp_loss[1].item(), phase="train")
        return train_loss

    def validation_loop(self, ):
        self.model.eval()
        fsdp_loss = torch.zeros(2).to(self.local_rank)
        inner_pbar = tqdm.tqdm(
            range(len(self.eval_dataloader)), colour="green", desc="Validation Epoch", disable=self.rank != 0
        )
        if self.rank == 0:
            start_time = time.time()
        with torch.no_grad():
            for batch in self.eval_dataloader:
                _ = batch.pop("dialog_ids", [])
                for key in batch.keys():
                    batch[key] = batch[key].to(self.local_rank)
                output = self.model(**batch)
                loss = output["loss"].detach()

                loss = self.cleanup_loss(loss, fsdp_loss)

                fsdp_loss[0] += loss  # sum up batch loss
                fsdp_loss[1] += 1  # sum up batch count

                inner_pbar.update(1)

        dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
        val_loss = fsdp_loss[0] / fsdp_loss[1]
        inner_pbar.close()
        self.print(f"--> validation Loss: {val_loss:.4f}")
        # report metrics to wandb
        if self.args.report_to_wandb and self.rank == 0:
            self._log(start_time, fsdp_loss[0].item(), fsdp_loss[1].item(), phase="eval")
        return val_loss

    def training_step(self, batch):
        for key in batch.keys():
            batch[key] = batch[key].to(self.local_rank)

        output = self.model(**batch)
        loss = output["loss"]
        # Do grad accumulation
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()

        return loss.detach()

    def cleanup_loss(self, loss, fsdp_loss):
        # Add loss to fsdp_loss and if it is nan or inf add the average
        loss_to_add = loss.item()
        if torch.isnan(loss) or torch.isinf(loss):
            loss_to_add = fsdp_loss[0] / fsdp_loss[1]
            if fsdp_loss[1] == 0:
                loss_to_add = 1.0
        return loss_to_add

    def gradient_accumulation(self):
        # with torch.set_grad_enabled(True):
        # Perform gradient acc if is multiple of gradient_acc_steps or if is last step in epoch
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
        # Updating parameters
        self.optimizer.step()
        self.scheduler.step()
        # Clear gradients w.r.t. parameters
        self.optimizer.zero_grad()

    def save_checkpoint(self, checkpoint_name=None):
        self.print(f"----> entering save model state")

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
                self.model, StateDictType.FULL_STATE_DICT, save_policy
        ):
            cpu_state = self.model.state_dict()

        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{self.global_step}"

        if self.rank == 0:
            self.print(f"----> saving model ...")
            self.print(f"----> saving model with name {checkpoint_name}")
            # Check if number of save surpasses the limit
            if len(self.saved_checkpoints) >= self.args.save_total_limit:
                # Get the oldest folder
                dir_to_delete = os.path.join(self.args.output_dir, self.saved_checkpoints[0])
                # Delete the folder
                print(f"----> deleting folder {dir_to_delete}")
                shutil.rmtree(dir_to_delete)
                # Update checkpoint list
                self.saved_checkpoints = self.saved_checkpoints[1:]
            # torch.save(cpu_state, os.path.join(self.args.output_dir, save_name))
            # https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict
            self.model.save_pretrained(os.path.join(self.args.output_dir, checkpoint_name), state_dict=cpu_state)
            self.print("----> saving tokenizer")
            self.tokenizer.save_pretrained(os.path.join(self.args.output_dir, checkpoint_name))
            if self.args.save_optimizer:
                self.print("----> saving optimizer")
                optim_state_dict = {'optim_state_dict': self.optimizer.state_dict()}
                torch.save(optim_state_dict, os.path.join(self.args.output_dir, checkpoint_name, "optim_state.pkl"))
            # added to list of saved checkpoints
            self.saved_checkpoints.append(checkpoint_name)
            self.print(f"----> finished saving model and tokenizer")
        return checkpoint_name

    def _log(self, start_time, total_loss, elapsed_steps, phase="train"):
        metrics = {"epoch": self.global_step / self.batches_per_epoch,
                   "sec_per_batch": (time.time() - start_time) / elapsed_steps,
                   "loss": total_loss / elapsed_steps}
        metrics["PPL"] = math.exp(metrics["loss"])

        if not (phase.lower() == "eval" or self.scheduler is None):
            metrics["learning_rate"] = self.scheduler.get_last_lr()[0]

        self.print(phase + " " + str(metrics))

        if self.args.report_to_wandb:
            def name(prefix, k):
                if k == "epoch":
                    return prefix.split("/")[0] + "/" + k
                return prefix + k

            prefix = self.args.run_type + "/" + phase + "_"

            self._wandb_log({name(prefix, k): v for k, v in metrics.items()})

    def _wandb_log(self, metrics: dict):
        wandb.log(metrics, step=self.global_step)

    def print(self, out):
        if self.rank == 0:
            print(out)


class PerpetualFSDPTrainer(FSDPTrainer):

    def __init__(self,
                 model: PreTrainedModel,
                 local_rank: int,
                 rank: int,
                 world_size: int,
                 tokenizer: PreTrainedTokenizer,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 optimizer: Optimizer = None,
                 args: TrainArgs = None,
                 data_args: DataArguments = None):

        if args.max_steps < 0:
            args.max_steps = 10000
        super().__init__(model, local_rank, rank, world_size, tokenizer, train_dataloader, eval_dataloader, optimizer, args, data_args)

        if self.rank == 0:
            self.print("Saving mutable parameters...")
            mutable_params = {
                "save_steps": self.args.save_steps,
                "save_total_limit": self.args.save_total_limit,
                "save_strategy": self.args.save_strategy,
                "evaluation_strategy": self.args.evaluation_strategy,
                "eval_steps": self.args.eval_steps,
                "logging_steps": self.args.logging_steps,
                "dataset_name": self.data_args.dataset_name,
                "learning_rate": self.args.learning_rate,
                "lr_scheduler_type": self.args.lr_scheduler_type,
                "max_steps": self.args.max_steps,
                "should_end": False,
            }

            with open(os.path.join(self.args.output_dir, "mutable_params.json"), "w") as f:
                json.dump(mutable_params, f)
        dist.barrier()

    def train(self):
        # TODO add perpetual training support
        self.print(" +++++++++++++++++++++++ RUN STATISTICS +++++++++++++++++++++++")
        self.print(f"Epochs: {self.num_train_epochs}")
        self.print(f"Batch Size: {self.args.per_device_train_batch_size}")
        self.print(f"Batches per Epoch: {self.batches_per_epoch}")
        self.print(f"Evaluation Batches: {self.eval_batches}")
        self.print(f"Warmup Steps: {self.warmup_steps}")
        self.print(f"Total Train Steps: {self.total_steps}")
        self.print(" ++++++++++++++++++++++++++ TRAINING ++++++++++++++++++++++++++")

        callback = None
        if self.args.param_update_strategy == CustomIntervalStrategy.STEPS:
            callback = functools.partial(self.read_instructions)

        should_end = False
        epoch_counter = 0
        while not should_end:
            should_end = self.read_instructions(can_update_data=True)
            if should_end:
                self.print("Read should end from mutable params file, stopping training...")
                break
            self.print(f"Epoch {epoch_counter} starting...")
            self.print("--> entering train loop")
            _ = self.train_loop(epoch_counter, callback)

            self.print(f"--> train done")

            if self.args.evaluation_strategy == CustomIntervalStrategy.EPOCH:
                self.print(f"--> running validation for epoch {epoch_counter}")
                eval_loss = self.validation_loop()
                if self.args.save_strategy == CustomIntervalStrategy.BEST_EVAL and self.best_val_loss > eval_loss:
                    self.best_val_loss = eval_loss
                    self.print(f"--> saving best model with loss: {self.best_val_loss}")
                    self.save_checkpoint()
            if self.args.save_strategy == CustomIntervalStrategy.EPOCH:
                self.print(f"--> saving on epoch {epoch_counter}")
                self.save_checkpoint()
            self.print(f"--> epoch {epoch_counter} completed")

            if self.global_step >= self.total_steps:
                self.print("----> global step >= total steps, stopping training...")
                should_end = True

            epoch_counter += 1
            dist.barrier()
        dist.barrier()

    def read_instructions(self, can_update_data=False):
        with open(os.path.join(self.args.output_dir, "mutable_params.json"), "r") as f:
            instructions = json.load(f)

        update_scheduler = False
        if instructions["save_strategy"] != self.args.save_strategy:
            self.args.save_strategy = instructions["save_strategy"]
            self.print(f"----> updating save strategy to {self.args.save_strategy}")
        if instructions["save_steps"] != self.args.save_steps:
            self.args.save_steps = instructions["save_steps"]
            self.print(f"----> updating save steps to {self.args.save_steps}")
        if instructions["save_total_limit"] != self.args.save_total_limit:
            self.args.save_total_limit = instructions["save_total_limit"]
            self.print(f"----> updating save total limit to {self.args.save_total_limit}")
        if instructions["evaluation_strategy"] != self.args.evaluation_strategy:
            self.args.evaluation_strategy = instructions["evaluation_strategy"]
            self.print(f"----> updating evaluation strategy to {self.args.evaluation_strategy}")
        if instructions["eval_steps"] != self.args.eval_steps:
            self.args.eval_steps = instructions["eval_steps"]
            self.print(f"----> updating eval steps to {self.args.eval_steps}")
        if instructions["logging_steps"] != self.args.logging_steps:
            self.args.logging_steps = instructions["logging_steps"]
            self.print(f"----> updating logging steps to {self.args.logging_steps}")
        if can_update_data and instructions['dataset_name'] != self.data_args.dataset_name:
            self.data_args.dataset_name = instructions['dataset_name']
            # TODO call function to update dataset and related metrics
            self.forget_dataloaders()
            self.load_data()
            self.print(f"----> updating dataset name to {self.data_args.dataset_name}")
        if instructions['learning_rate'] != self.args.learning_rate:
            self.args.learning_rate = instructions['learning_rate']
            self.print(f"----> updating learning rate to {self.args.learning_rate}")
            update_scheduler = True
        if instructions['lr_scheduler_type'] != self.args.lr_scheduler_type:
            self.args.lr_scheduler_type = instructions['lr_scheduler_type']
            self.print(f"----> updating lr scheduler type to {self.args.lr_scheduler_type}")
            update_scheduler = True
        if instructions['max_steps'] != self.args.max_steps:
            self.args.max_steps = instructions['max_steps']
            self.total_steps = self.args.max_steps
            self.print(f"----> updating max steps to {self.args.max_steps}")
        should_end = bool(instructions["should_end"])

        if update_scheduler:
            self.update_scheduler()
            self.print(f"----> updating scheduler")

        # wait for all processes to finish reading instructions and updating args
        dist.barrier()
        return should_end

    def forget_dataloaders(self):
        del self.eval_dataloader
        del self.train_dataloader
        self.train_dataloader = None
        self.eval_dataloader = None
        self.batches_per_epoch = 0
        self.eval_batches = 0
        self.warmup_steps = 0

    def load_data(self):
        self.train_dataloader, self.eval_dataloader = data_utils.load_data(self.tokenizer, self.data_args, self.args, init_process_group=False)
        self.batches_per_epoch = (len(self.train_dataloader) // self.args.gradient_accumulation_steps)
        self.batches_per_epoch = max(self.batches_per_epoch, 1)

        self.eval_batches = len(self.eval_dataloader)

        self.warmup_steps = self.args.warmup_steps if self.args.warmup_steps >= 0 else int(
            self.batches_per_epoch * self.args.warmup_ratio)

    def update_scheduler(self):
        if self.args.lr_scheduler_type == CustomSchedulerType.CYCLICAL or self.args.lr_scheduler_type == "cyclical":
            self.scheduler = trainer_utils.get_cyclical_schedule(self.optimizer, self.args.learning_rate,
                                                                 self.args.lr_step_size)
        else:
            self.scheduler = transformers.get_scheduler(self.args.lr_scheduler_type, self.optimizer,
                                                        self.warmup_steps, self.total_steps)

