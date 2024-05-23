import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from typing import Callable, Optional

class StableDiffusion:
    def __init__(self) -> None:
        self.pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1",
                                            torch_dtype=torch.float16,
                                            requires_safety_checker=False,
                                            ).to("cuda")

        self.pipeline.tokenizer.truncation_side = 'left'

        negative_prompt_array = ["hands", "human", "person", "cropped", "deformed", "cut off", "malformed", "out of frame", "split image", "tiling", "watermark", "text"]

        self.negative_prompt = ", ".join(negative_prompt_array)
        self.use_negative_prompt = True


    def call(self, prompt: str, latents: Optional[torch.FloatTensor] = None, callback: Optional[Callable] = None, callback_steps=1, generator = None):
        negative_prompt = self.negative_prompt if self.use_negative_prompt else None
        return self.pipeline(prompt, latents=latents, negative_prompt=negative_prompt, callback=callback, callback_steps=callback_steps, generator=generator).images[0]


    def load_i2i(self):
        self.pipeline_i2i = StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1",
                                            torch_dtype=torch.float16,
                                            requires_safety_checker=False,
                                            ).to("cuda")


    def call_i2i(self, prompt: str, image, strength=0.75, guidance_scale=7.5):
        negative_prompt = self.negative_prompt if self.use_negative_prompt else None
        return self.pipeline_i2i(prompt, image=image, negative_prompt=negative_prompt, strength=strength, guidance_scale=guidance_scale).images[0]


    def set_use_negative_prompts(self, value: bool):
        self.use_negative_prompt = value