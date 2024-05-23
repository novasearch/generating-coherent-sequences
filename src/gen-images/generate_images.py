import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

from dotenv import load_dotenv
import os
import boto3
from io import BytesIO
from tqdm import tqdm

from PIL import Image

load_dotenv()

model_id = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
# i2i_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

pipe.tokenizer.truncation_side = 'left'

negative_prompts = ["hands", "human", "person", "cropped", "deformed", "cut off", "malformed", "out of frame", "split image", "tiling", "watermark", "text"]

negative_prompt = ", ".join(negative_prompts)

EXTRA_ARGS = {
    "ContentType": "image/png",
    "ACL": "public-read",
}

def split_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def expand_list(lst, n):
    return [num for num in lst for _ in range(n)]


def gen_images(prompts_, images_folder, prompt_type, num_images_per_prompt=4, num_images_per_batch=4, upload=False):
    assert num_images_per_batch % num_images_per_prompt == 0 and num_images_per_batch >= num_images_per_prompt

    ids = []
    prompts = []

    for prompt in prompts_:
        if not prompt["prompt"]:
            continue
        ids.append(prompt["id"])
        prompts.append(prompt["prompt"])

    prompts_chunks = split_list(prompts, chunk_size=int(num_images_per_batch / num_images_per_prompt))
    ids_chunks = split_list(ids, chunk_size=int(num_images_per_batch / num_images_per_prompt))

    for prompts, ids in tqdm(zip(prompts_chunks, ids_chunks), total=len(prompts_chunks), desc="Generating images"):
        images = pipe(prompts, num_images_per_prompt=num_images_per_prompt, negative_prompt=[negative_prompt]).images
        paths_exp = expand_list(ids, num_images_per_prompt)
        for i, (image, image_id) in enumerate(zip(images, paths_exp)):
            # image.save(f"{images_folder}/{image_id}_{i % num_images_per_prompt}.png")
            image.save(f"{images_folder}/{image_id}.png")


# def gen_image_to_image(prompt_, previous_prompt_id, images_folder, num_images_per_prompt=4, strength=0.80, guidance_scale=7.5):
#     id = prompt_["id"]
#     prompt = prompt_["prompt"]

#     images = []
#     for i in range(num_images_per_prompt):
#         # Try Image-to-Image
#         try:
#             if not previous_prompt_id:
#                 raise FileNotFoundError

#             previous_image = Image.open(f"{images_folder}/{previous_prompt_id}_{i}.png")

#             print("Image to Image")
#             images = i2i_pipe(prompt=prompt, image=previous_image, strength=strength, guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt).images

#         # Resort to individual generation
#         except FileNotFoundError:
#             print("Text to Image")
#             images = pipe(prompt, num_images_per_prompt=num_images_per_prompt).images

#     for i, image in enumerate(images):
#         image.save(f"{images_folder}/{id}_{i}.png")