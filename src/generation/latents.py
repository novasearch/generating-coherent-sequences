import torch
from typing import Optional, List

class Latents:
    def __init__(self) -> None:
        self.history : List[torch.FloatTensor] = []


    def __str__(self) -> Optional[str]:
        if self.is_empty():
            return None
        
        return str(len(self.history))


    def is_empty(self) -> bool:
        return self.latents is None


    def add_latents(self, latents: torch.FloatTensor):
        self.history.append(latents)


    def get_latents(self, diffusion_step: int) -> torch.FloatTensor:
        return self.history[diffusion_step]


    def get_history(self) -> List[torch.FloatTensor]:
        return self.history