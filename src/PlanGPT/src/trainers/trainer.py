import math
import os
import shutil
import subprocess
import time
from typing import Union, Any, Mapping, Dict, Tuple

import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import PrecisionType

from data_mod import data_utils
from data_binding import TrainArgs, CustomSchedulerType, CustomIntervalStrategy, DataArguments
from torch import Tensor, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, PreTrainedModel, PreTrainedTokenizer

import trainers.trainer_utils as trainer_utils


def _prepare_input(data: Union[torch.Tensor, Any], device) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device=device)
        return data.to(**kwargs)
    return data


def _prepare_inputs(inputs: Dict[str, Union[torch.Tensor, Any]], device) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    inputs = _prepare_input(inputs, device)

    return inputs


class LabTrainer:
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 optimizer: Optimizer = None,
                 args: TrainArgs = None,
                 data_args: DataArguments = None):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.args = args
        self.optimizer = optimizer
        self.data_args = data_args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

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

        self.eval_batches = 0
        if self.args.evaluation_strategy != CustomIntervalStrategy.NO:
            self.eval_batches = len(eval_dataloader)

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

        if self.args.infer_checkpoints:
            self.eval_file = os.path.join(self.data_args.data_path, self.data_args.dataset_name, self.data_args.dataset_name + "_eval.json")

        if args.report_to_wandb:
            self._init_wandb()

    def _init_wandb(self):
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
                if self.args.infer_checkpoints and self.is_main_process():
                    subprocess.call(['sbatch', 'scripts/infer_checkpoint.sh', os.path.join(self.args.output_dir, ckpt_name),
                                     self.eval_file])
            self.print(f"--> epoch {epoch} completed")

    def train_loop(self, epoch, callback=None):
        self.model.train()
        fsdp_loss = torch.zeros(2).to(self.device)

        inner_pbar = tqdm(range(self.batches_per_epoch), colour="blue", desc=f"Training Epoch {epoch}")

        start_step = self.global_step
        start_time = time.time()

        step_loss = 0.0

        for step, batch in enumerate(self.train_dataloader):
            with torch.set_grad_enabled(True):

                loss = self.training_step(batch)

                loss = self.cleanup_loss(loss, fsdp_loss)

                step_loss += loss

                if (step + 1) % self.args.gradient_accumulation_steps == 0 \
                        or step == self.batches_per_epoch - 1:
                    fsdp_loss[0] += step_loss
                    fsdp_loss[1] += 1
                    self.global_step += 1
                    self.gradient_accumulation()
                    inner_pbar.update(1)
                    if self.args.report_to_wandb and self.global_step % self.args.logging_steps == 0:
                        self._log(start_time, fsdp_loss[0].item(), self.global_step - start_step, phase="train", step_loss=step_loss)
                    if self.args.evaluation_strategy == CustomIntervalStrategy.STEPS and self.global_step % self.args.eval_steps == 0:
                        self.print(f"--> running validation@{self.global_step}")
                        _ = self.validation_loop()
                        self.model.train()
                    if self.args.save_strategy == CustomIntervalStrategy.STEPS and self.global_step % self.args.save_steps == 0:
                        self.print(f"--> saving model@{self.global_step}")
                        ckpt_name = self.save_checkpoint()
                        if self.args.infer_checkpoints and self.global_step >= self.args.warmup_before_inference and self.is_main_process():
                            subprocess.call(['sbatch', 'scripts/infer_checkpoint.sh',
                                             os.path.join(self.args.output_dir, ckpt_name),
                                             self.eval_file])

                    step_loss = 0.0

                if callback is not None and self.global_step % self.args.param_update_interval == 0 and self.args.param_update_strategy == CustomIntervalStrategy.STEPS:
                    self.print(f"--> updating params@{self.global_step}")
                    callback(self)

                if self.global_step == self.total_steps:
                    self.print(f"--> stopping training@{self.global_step}")
                    break
        # self.print(f"Fsdp_loss: {fsdp_loss} avg: {fsdp_loss[0].item() / fsdp_loss[1].item()}")
        train_loss = fsdp_loss[0].item() / (self.global_step - start_step)

        inner_pbar.close()
        self.print(
            f"--> train epoch: \t{epoch}, loss: \t{train_loss:.4f}"
        )
        # report metrics to wandb
        if self.args.report_to_wandb:
            self._log(start_time, fsdp_loss[0].item(), (self.global_step - start_step), phase="train")
        return train_loss

    def validation_loop(self, ):
        self.model.eval()
        fsdp_loss = torch.zeros(2).to(self.device)
        inner_pbar = tqdm(range(len(self.eval_dataloader)), colour="green", desc="Validation Epoch")

        start_time = time.time()
        with torch.no_grad():
            for batch in self.eval_dataloader:
                _ = batch.pop("dialog_ids", [])
                _ = batch.pop("labels_mask", [])
                _ = batch.pop("negative_labels", [])
                _ = batch.pop("negative_labels_mask", [])
                _ = batch.pop("input_ids_and_negative_labels", [])
                for key in batch.keys():
                    batch[key] = batch[key].to(self.device)
                output = self.model(**batch)
                loss = output["loss"].detach()

                loss = self.cleanup_loss(loss, fsdp_loss)

                fsdp_loss[0] += loss  # sum up batch loss
                fsdp_loss[1] += 1  # sum up batch count

                inner_pbar.update(1)

        val_loss = fsdp_loss[0].item() / fsdp_loss[1].item()
        inner_pbar.close()
        self.print(f"--> validation Loss: {val_loss:.4f}")
        # report metrics to wandb
        if self.args.report_to_wandb:
            self._log(start_time, fsdp_loss[0].item(), fsdp_loss[1].item(), phase="eval")
        return val_loss

    def training_step(self, batch):
        _ = batch.pop("dialog_ids", [])
        _ = batch.pop("labels_mask", [])
        _ = batch.pop("negative_labels", [])
        _ = batch.pop("negative_labels_mask", [])
        _ = batch.pop("input_ids_and_negative_labels", [])
        for key in batch.keys():
            batch[key] = batch[key].to(self.device)

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

        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{self.global_step}"

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

        trainer_utils.save_model(self.model, os.path.join(self.args.output_dir, checkpoint_name), self.optimizer, self.tokenizer)
        self.print(f"----> finished saving model and tokenizer")
        return checkpoint_name

    def _log(self, start_time, total_loss, elapsed_steps, phase="train", step_loss=None):
        avg_loss = total_loss / elapsed_steps
        loss = step_loss if step_loss is not None else avg_loss

        metrics = {"epoch": self.global_step / self.batches_per_epoch,
                   "sec_per_batch": (time.time() - start_time) / elapsed_steps,
                   "loss": loss,
                   "avg_loss": avg_loss,}
        # this line ocasionally causes an OverflowError: math range error
        # need to wrap it in a try/except

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
        print(out)

    def is_main_process(self):
        return True


class AccelTrainer(LabTrainer):
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 optimizer: Optimizer = None,
                 args: TrainArgs = None):
        self.accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                                       log_with=["wandb"] if args.report_to_wandb else None,
                                       mixed_precision=PrecisionType.BF16 if args.mixed_precision == PrecisionType.BF16.value else PrecisionType.FP16 if args.mixed_precision == PrecisionType.FP16.value else None)

        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.args = args

        if self.accelerator.is_main_process and not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir, exist_ok=True)
        if self.accelerator.is_main_process and not os.path.exists(self.args.output_dir + "/ckpts"):
            os.makedirs(self.args.output_dir + "/ckpts", exist_ok=True)

        if self.args.report_to_wandb:
            self._init_wandb()

        self.model = self.accelerator.prepare(model)
        self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(train_dataloader, eval_dataloader)

        # METRIC CALCULATION
        self.batches_per_epoch = (len(train_dataloader) // self.args.gradient_accumulation_steps) // torch.cuda.device_count()
        self.batches_per_epoch = max(self.batches_per_epoch, 1)

        if args.max_steps > 0:
            self.total_steps = args.max_steps
            self.num_train_epochs = args.max_steps // self.batches_per_epoch + int(
                args.max_steps % self.batches_per_epoch > 0
            )
        else:
            self.total_steps = math.ceil(args.num_train_epochs * self.batches_per_epoch)
            self.num_train_epochs = math.ceil(args.num_train_epochs)

        self.eval_batches = len(eval_dataloader)

        self.warmup_steps = args.warmup_steps if args.warmup_steps >= 0 else int(self.total_steps * 0.02)

        self.saved_checkpoints = []

        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_epsilon,
            )

        if self.args.lr_scheduler_type == CustomSchedulerType.CYCLICAL or self.args.lr_scheduler_type == "cyclical":
            self.scheduler = trainer_utils.get_cyclical_schedule(self.optimizer, self.args.learning_rate,
                                                                 self.args.lr_step_size)
        else:
            self.scheduler = transformers.get_scheduler(self.args.lr_scheduler_type, self.optimizer,
                                                        self.args.warmup_steps, self.total_steps)

        self.optimizer, self.scheduler = self.accelerator.prepare(self.optimizer, self.scheduler)

        self._best_eval_loss = 10000000000000.0
        self._best_eval_epoch = -1
        self.global_step = 0

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        self.logging_steps = max(args.logging_steps, 1)

        self.step_func = self._hf_step

    def train(self):
        self.accelerator.wait_for_everyone()
        super(AccelTrainer, self).train()

    def train_epoch(self, epoch: float):
        self.model.train()
        total_loss = 0.0
        epoch_start_time = time.time()
        last_logged_step = 0
        last_log_time = time.time()
        initial_step = self.global_step
        progress_bar = tqdm(total=self.batches_per_epoch, desc="Training", disable=not self.accelerator.is_main_process)
        for step, batch in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.model):
                # self.print(step)
                inputs = batch
                loss = self.training_step(step, inputs)

                total_loss += loss

                if (step % self.args.gradient_accumulation_steps == 0 and step > 0) or step == len(self.train_dataloader)-1:
                    progress_bar.update(1)
                    self.global_step += 1
                    # print(f"Device: {torch.cuda.current_device()}, global step: {self.global_step}")
                    if self.global_step % self.logging_steps == 0:
                        self.print("logging....")
                        self._log(last_log_time, total_loss, elapsed_steps=self.logging_steps, phase="train")
                        last_log_time = time.time()
                    if self.args.save_strategy == CustomIntervalStrategy.STEPS and self.global_step % self.args.save_steps == 0:
                        self.print(f"Saving model at step {self.global_step}")
                        save_path = f"{self.args.output_dir}/ckpts/checkpoint_{self.global_step}"
                        self.save_model(save_path, is_checkpoint=True)
                        if self.args.infer_checkpoints and self.accelerator.is_main_process:
                            subprocess.call(['sbatch', 'scripts/infer_checkpoint.sh', save_path])

                if self.global_step == self.total_steps:
                    break
        progress_bar.close()
        print(f"Device: {torch.cuda.current_device()}, Finished epoch: {epoch}")
        self._log(epoch_start_time, total_loss, elapsed_steps=(self.global_step - initial_step), phase="train")

    def training_step(self, inputs):
        loss, _ = self.step_func(inputs)

        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), 3.0)

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return loss.item()

    def _hf_step(self, inputs) -> Tuple[Tensor, Dict]:
        outputs = self.model(**inputs)
        loss = outputs["loss"]
        return loss, outputs

    def evaluate(self):
        self.accelerator.wait_for_everyone()
        self.model.eval()
        total_loss = 0.0
        step_count = 0
        for step, batch in tqdm(enumerate(self.eval_dataloader)):
            with torch.no_grad():
                loss, _ = self.step_func(batch)
                # https://huggingface.co/docs/accelerate/v0.17.1/en/basic_tutorials/notebook#writing-the-training-function
                losses = self.accelerator.gather(loss)
                step_count += losses.shape[0]
                total_loss += losses.sum()

        eval_loss = total_loss / step_count
        self.print(f" Evaluation step count: {step_count}")
        # TODO see if it makes sense to ignore predictions on pad tokens
        # TODO see line 42 https://github.com/affjljoo3581/GPT2/blob/master/src/gpt2/evaluate_model.py
        eval_ppl = torch.exp(torch.tensor(eval_loss))
        return eval_loss, total_loss.item(), eval_ppl

    def _init_wandb(self):
        # TODO Check if this is working as intended
        # TODO This shouldn't be here, because if I do pretrain then finetune it messes up the wandb reporting
        self.accelerator.init_trackers(project_name=self.args.project_name, config=self.args.to_dict())
        if self.accelerator.is_main_process:
            self.accelerator.get_tracker("wandb").name = self.args.project_name

    def print(self, output):
        self.accelerator.print(output)

    def save_model(self, save_dir: str = None, is_checkpoint=False):
        # Cannot use the self when defining default values so this has to be done
        if save_dir is None:
            save_dir = self.args.output_dir

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            # Save model state and config (needed so that we can then use .from_pretrained when loading the model)
            # self.accelerator.unwrap_model(self.model).save_pretrained(save_dir)
            if is_checkpoint:
                ckpt_path = "/".join(save_dir.split("/")[:-1])

                # Check if number of save surpasses the limit
                if "ckpts" in save_dir and len(self.saved_checkpoints) >= self.args.save_total_limit:
                    # Get the oldest folder
                    dir_to_delete = ckpt_path + "/" + self.saved_checkpoints[0]
                    # Delete the folder
                    print(f"Deleting folder {dir_to_delete}")
                    shutil.rmtree(dir_to_delete)
                    # Update checkpoint list
                    self.saved_checkpoints = self.saved_checkpoints[1:]

            trainer_utils.save_model(self.accelerator.unwrap_model(self.model), save_dir, self.optimizer,
                                     self.tokenizer, state_dict=self.accelerator.get_state_dict(self.model))

            if is_checkpoint:
                ckpt_name = save_dir.split("/")[-1]
                self.saved_checkpoints.append(ckpt_name)

    def _wandb_log(self, metrics: dict):
        self.accelerator.log(metrics, step=self.global_step)
