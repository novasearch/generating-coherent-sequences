import os
import time

from trainers.trainer import LabTrainer, _prepare_inputs
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
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup, PreTrainedModel, PreTrainedTokenizer

import copy


class DPOTrainer(LabTrainer):

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 optimizer: Optimizer = None,
                 args: TrainArgs = None,
                 data_args: DataArguments = None):

        super(DPOTrainer, self).__init__(model, tokenizer, train_dataloader, eval_dataloader, optimizer, args,
                                         data_args)

        print(self.model.device)

        self.print_mem_stats(3)

        self.use_original = True

        print("PEFT Model Config: ", self.model.config)

    def training_step(self, inputs):
        # self.check_if_params_updated()
        with torch.autograd.set_detect_anomaly(True):
            with torch.set_grad_enabled(True):
                self.print_mem_stats(4)
                for key in inputs.keys():
                    inputs[key] = inputs[key].to(self.device)

                prep_inputs = _prepare_inputs(inputs, self.device)

                negative_labels = prep_inputs.pop("negative_labels", None)
                labels_mask = prep_inputs.pop("labels_mask", None)
                negative_labels_mask = prep_inputs.pop("negative_labels_mask", None)
                input_ids_and_negative_labels = prep_inputs.pop("input_ids_and_negative_labels", None)

                self.print_mem_stats(5)
                outputs = self.model(**prep_inputs)
                ce_loss = outputs["loss"]
                # ce_loss.backward()
                # print("cross_entropy_loss: ", ce_loss.item())
                self.print_mem_stats(6)
                # if self.use_original:
                # loss, _ = self.compute_loss(prep_inputs, outputs, negative_labels, labels_mask, negative_labels_mask)
                # print("Original Loss: ", loss.item())
                # else:
                if self.args.interval_dpo:
                    loss = ce_loss
                    if self.global_step % self.args.interval_dpo_steps == 0:
                        neg_outputs = self.model(input_ids=input_ids_and_negative_labels, labels=negative_labels)
                        dpo_loss, _, _ = self.compute_loss_v2(prep_inputs, outputs, neg_outputs,
                                                              input_ids_and_negative_labels, negative_labels,
                                                              labels_mask, negative_labels_mask)
                        loss = dpo_loss
                else:
                    neg_outputs = self.model(input_ids=input_ids_and_negative_labels, labels=negative_labels)
                    dpo_loss, _, _ = self.compute_loss_v2(prep_inputs, outputs, neg_outputs, input_ids_and_negative_labels, negative_labels, labels_mask, negative_labels_mask)
                    # print("Modified Loss: ", dpo_loss.item())

                    # print("dpo_loss: ", loss.item())
                    self.print_mem_stats(7)

                    if self.args.dpo_mixed:
                        loss = dpo_loss + self.args.dpo_theta * ce_loss
                    else:
                        loss = dpo_loss

                # Do grad accumulation
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                self._wandb_log({"ce_loss: ": ce_loss.item()})

                self.print_mem_stats(8)
            return loss

    def compute_loss(self, inputs, outputs, negative_labels, labels_mask, negative_labels_mask):
        start_time = time.time()
        # print("inputs: ", inputs.keys())
        beta = self.args.dpo_beta

        pi_logs = F.log_softmax(outputs['logits'], dim=-1)
        print("pi_logs: ", sum(pi_logs[0][0]))

        with torch.no_grad() and self.model.disable_adapter():
            # get the log probabilities of the reference model
            ref_logs = F.log_softmax(self.model(**inputs)['logits'], dim=-1)

        inputs['labels'][~labels_mask] = self.tokenizer.pad_token_id
        labels = inputs['labels']
        negative_labels[~negative_labels_mask] = self.tokenizer.pad_token_id

        pi_yw_probs = torch.gather(pi_logs, 2, labels.unsqueeze(2)).squeeze(-1)[labels_mask].mean(-1)
        pi_yl_probs = torch.gather(pi_logs, 2, negative_labels.unsqueeze(2)).squeeze(-1)[negative_labels_mask].mean(-1)
        ref_yw_probs = torch.gather(ref_logs, 2, labels.unsqueeze(2)).squeeze(-1)[labels_mask].mean(-1)
        ref_yl_probs = torch.gather(ref_logs, 2, negative_labels.unsqueeze(2)).squeeze(-1)[negative_labels_mask].mean(-1)

        self.print_mem_stats(6.75)

        pi_logratios = pi_yw_probs - pi_yl_probs
        ref_logratios = ref_yw_probs - ref_yl_probs

        ratios = (pi_logratios - ref_logratios)

        mid = beta * ratios

        losses = -F.logsigmoid(mid)

        rewards = beta * (pi_logs - ref_logs).detach()

        return losses, rewards.mean(-1)

    def compute_loss_v2(self, inputs, pos_outputs, neg_outputs, input_ids_and_negative_labels, negative_labels, labels_mask, negative_labels_mask):
        start_time = time.time()
        # print("inputs: ", inputs.keys())
        beta = self.args.dpo_beta

        pos_pi_logs = F.log_softmax(pos_outputs['logits'], dim=-1)
        neg_pi_logs = F.log_softmax(neg_outputs['logits'], dim=-1)

        with torch.no_grad() and self.model.disable_adapter():
            # get the log probabilities of the reference model
            pos_ref_logs = F.log_softmax(self.model(**inputs)['logits'], dim=-1)
            neg_ref_logs = F.log_softmax(self.model(input_ids=input_ids_and_negative_labels, labels=negative_labels)['logits'], dim=-1)

        inputs['labels'][~labels_mask] = self.tokenizer.pad_token_id
        labels = inputs['labels']
        negative_labels[~negative_labels_mask] = self.tokenizer.pad_token_id

        # pi_yw_probs = sum(torch.gather(pos_pi_logs, 2, labels.unsqueeze(2)).squeeze(-1)[labels_mask])
        # pi_yl_probs = sum(torch.gather(neg_pi_logs, 2, negative_labels.unsqueeze(2)).squeeze(-1)[negative_labels_mask])
        # ref_yw_probs = sum(torch.gather(pos_ref_logs, 2, labels.unsqueeze(2)).squeeze(-1)[labels_mask])
        # ref_yl_probs = sum(torch.gather(neg_ref_logs, 2, negative_labels.unsqueeze(2)).squeeze(-1)[negative_labels_mask])

        pi_yw_probs = torch.gather(pos_pi_logs, 2, labels.unsqueeze(2)).squeeze(-1)[labels_mask].mean(-1)
        pi_yl_probs = torch.gather(neg_pi_logs, 2, negative_labels.unsqueeze(2)).squeeze(-1)[negative_labels_mask].mean(-1)
        ref_yw_probs = torch.gather(pos_ref_logs, 2, labels.unsqueeze(2)).squeeze(-1)[labels_mask].mean(-1)
        ref_yl_probs = torch.gather(neg_ref_logs, 2, negative_labels.unsqueeze(2)).squeeze(-1)[negative_labels_mask].mean(-1)

        pi_logratios = pi_yw_probs - pi_yl_probs
        ref_logratios = ref_yw_probs - ref_yl_probs

        ratios = (pi_logratios - ref_logratios)

        mid = beta * ratios

        losses = -F.logsigmoid(mid)

        pos_rewards = beta * (pos_pi_logs - pos_ref_logs).detach()
        neg_rewards = beta * (neg_pi_logs - neg_ref_logs).detach()

        return losses, pos_rewards.mean(-1), neg_rewards.mean(-1)

    def print_mem_stats(self, idx):
        if False and torch.cuda.is_available() and self.model.device.type == 'cuda':
            local_rank = int(os.environ['LOCAL_RANK'])
            t = torch.cuda.get_device_properties(local_rank).total_memory / 1024 ** 3
            r = torch.cuda.memory_reserved(local_rank) / 1024 ** 3
            a = torch.cuda.memory_allocated(local_rank) / 1024 ** 3
            self.print("{} - Total Memory: {}".format(idx, t))
            self.print("{} - Reserved Memory: {}".format(idx, r))
            self.print("{} - Allocated Memory: {}".format(idx, a))
            self.print("{} - Free Memory: {}".format(idx, r - a))



