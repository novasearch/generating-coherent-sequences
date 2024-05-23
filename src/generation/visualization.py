import torch
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from typing import List

def display_images_with_text(images, steps, path):
    # Create a figure
    fig = plt.figure(figsize=(6, len(images)*6))

    for i, step in enumerate(steps):
        # Wrap the text
        title_text = textwrap.fill(step, width=40)

        # Add the first image for this step
        ax = fig.add_subplot(len(steps), 1, i + 1)
        ax.imshow(images[i])
        ax.set_title(title_text, fontsize=18, pad=20)
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.savefig(f"{path}.png")


def display_tensor_image(tensor, iteration):
    # Detach the tensor, move to CPU, convert to float32, convert to numpy, and squeeze
    img = tensor.detach().cpu().to(torch.float32).numpy().squeeze(0)

    # Transpose from [color_channels, height, width] to [height, width, color_channels]
    img = np.transpose(img, (1, 2, 0))

    # Normalize to [0,1] if it's a floating point image
    if img.dtype == np.float32: 
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # Create figure and axes
    _, ax = plt.subplots()

    ax.set_title(f"Image at iteration {iteration}")

    ax.imshow(img)

    ax.axis("off")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the plot
    plt.savefig(f"noise/{iteration}.png")


def display_tensor_grid(image_lists, steps: int, filename: str, image_size=(3, 3)):
    
    print("image_lists.shape", len(image_lists), len(image_lists[0]))
    image_lists = [l[0::steps] for l in image_lists]
    print("image_lists.shape", len(image_lists), len(image_lists[0]))

    # Number of rows and columns
    num_rows = len(image_lists)
    num_cols = len(image_lists[0])

    # Create a subplot of size num_rows x num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * image_size[0], num_rows * image_size[1]))

    # Go through each row
    for i, row in enumerate(image_lists):
        # Go through each column in the row
        for j, tensor in enumerate(row):
            # Move tensor to CPU, convert to float32, then to numpy and squeeze
            img = tensor.detach().cpu().to(torch.float32).numpy().squeeze(0)

            # Transpose from [channels, height, width] to [height, width, channels] 
            img = np.transpose(img, (1, 2, 0))

            # Normalize if it's a floating point image
            if img.dtype == np.float32:
                img = (img - np.min(img)) / (np.max(img) - np.min(img))

            # Show tensor as image in the corresponding subplot
            axs[i, j].imshow(img)
            
            axs[i, j].axis('off')

            if i == 0:
                axs[i, j].text(0.5, 1.1, str(j * steps), transform=axs[i, j].transAxes, fontsize=16, ha='center')

    # Automatically adjust subplot parameters to give specified padding
    plt.tight_layout()

    plt.savefig(f"{filename}.png")


def display_runs_side_by_side(images, steps, nmethods, path):
    nsteps = len(steps)
    #nmethods = int(len(images) / nsteps)

    print(f"nmethods: {nmethods}")

    fig, axes = plt.subplots(nrows=nsteps, ncols=(nmethods+1), figsize=(3*nmethods, 2.5*nsteps))
    r = fig.canvas.get_renderer()

    #for method in range(1, nmethods+1):
    #    axes[0, method].set_title(titles[method-1])

    axes[0, 0].set_title("Steps")
    #axes[0, 1].set_title(titles[1])

    # for i, arr in enumerate([steps, steps_prompt]):
    for step, step_text in enumerate(steps):
        # axes[step, i].axis('off')
        axes[step, 0].axis('off')
        txt = textwrap.fill(step_text, width=30)

        # tt = axes[step, i].text(1.0, 0.5 , txt,
        tt = axes[step, 0].text(1.0, 0.5 , txt,
            horizontalalignment='right',
            verticalalignment='center',
            bbox={'facecolor': 'none', 'alpha': 1, 'edgecolor': 'none', 'pad': 1},
            wrap = True)

    for m in range(0, nmethods):
        for s in range(0, nsteps):
            #axes[s,m+1].imshow(images[s+(m-1)*nsteps])
            axes[s,m+1].imshow(images[m]["images"][s])
            axes[0, m+1].set_title(images[m]["title"])

    for ax in axes.ravel():
        ax.set_xticklabels([], visible=False)
        ax.set_yticklabels([], visible=False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    fig.show()
    fig.savefig(f"{path}.png")


def display_images_and_tensors(I, L, steps, filename):
    print(len(I))
    print(len(L), len(L[0]))
    num_tensor_cols = 6 # Number of images per row for L
    num_images = len(I)

    # We need a tensor cols + real image + steps
    fig, axs = plt.subplots(num_images, num_tensor_cols + 1 + 1, figsize=(12, 3 * num_images))

    for step, step_text in enumerate(steps):
        # axes[step, i].axis('off')
        axs[step, 0].axis('off')
        txt = textwrap.fill(step_text, width=30)

        # tt = axes[step, i].text(1.0, 0.5 , txt,
        tt = axs[step, 0].text(1.0, 0.5 , txt,
            horizontalalignment='right',
            verticalalignment='center',
            bbox={'facecolor': 'none', 'alpha': 1, 'edgecolor': 'none', 'pad': 1},
            wrap = True)

    for i in range(num_images):
        
        # Display the actual image (+1 due to steps)
        axs[i, 0+1].imshow(I[i])
        axs[i, 0+1].axis('off')

        # rewrite L[i] to contain first 3 and last 3
        tensor_arr = L[i][:3] + L[i][-3:]

        tensor_arr.reverse()

        # Display tensors from corresponding arrays in L horizontally
        for j in range(num_tensor_cols):
            tensor = tensor_arr[j]
            img = tensor.detach().cpu().to(torch.float32).numpy().squeeze(0)
            img = np.transpose(img, (1, 2, 0))
            if img.dtype == np.float32:
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
            # +2 Due to steps and real image
            axs[i, j + 1 + 1].imshow(img)
            axs[i, j + 1 + 1].axis('off')

    plt.tight_layout()
    plt.savefig(f"{filename}.png")