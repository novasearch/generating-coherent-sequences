from transformers import GenerationConfig

MAX_TOKENS = 77

GREEDY_CONFIG = GenerationConfig(
    max_new_tokens=MAX_TOKENS,
)

BEAM_CONFIG = GenerationConfig(
    max_new_tokens=MAX_TOKENS,
    num_beams=8,
    early_stopping=True
)

SAMPLING_CONFIG = GenerationConfig(
    max_new_tokens=MAX_TOKENS,
    do_sample=True,
    temperature=0.6,
    top_k=0,
)

TOP_K_CONFIG = GenerationConfig(
    max_new_tokens=MAX_TOKENS,
    do_sample=True,
    temperature=0.6,
    top_k=50,
)

TOP_P_CONFIG = GenerationConfig(
    max_new_tokens=MAX_TOKENS,
    do_sample=True,
    temperature=0.6,
    top_p=0.92,
    top_k=0
)

TOP_PK_CONFIG = GenerationConfig(
    max_new_tokens=MAX_TOKENS,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    top_k=50
)