import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to('cpu')
weight_decay = 0.1
learning_rate = 1.0E-6
gradient_accumulation_steps = 4

batches_per_epoch = 5000
epochs = 3

total_steps = batches_per_epoch * epochs
warmup_steps = int((total_steps // gradient_accumulation_steps) * 0.05)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    betas=(0.9, 0.98),
    eps=1e-9
)

step_size = min(max(batches_per_epoch, total_steps // 20), total_steps // 10)

scheduler = CosineAnnealingWarmRestarts(optimizer, 10, 2)

lrs = []

for s in range(total_steps):
    scheduler.step()
    lrs.append(scheduler.get_last_lr()[0])

xpoints = np.array(np.arange(len(lrs)))
ypoints = np.array(lrs)

plt.plot(xpoints, ypoints)
plt.show()
