DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

ALPACA_PROMPT_DICT = (
    "Below is a recipe with instructions on how to make it, paired with an input that shows the dialog between you and a user about the recipe. "
    "Write a response that appropriately responds to the user request.\n\n"
    "### Recipe:\n{recipe}\n\n### Dialog:\n{dialog}\n\n### Response: "
)

OA_PROMPT_DICT = (
    "<|prompter|> I will give you a recipe and I want you to help me do it step by step. Please use a {system_tone} tone of voice. Recipe: {recipe} This is the current step: {current_step}. <|endoftext|> <|assistant|> ok! <|endoftext|> {dialog} <|endoftext|> <|assistant|>"
)

VICUNA_PROMPT_DICT = (
    "{user_token} I will give you a recipe and I want you to help me do it step by step. Please use a {system_tone} "
    "tone of voice. Recipe: {recipe} {current_step} {sep_token} {sys_token} ok! {sep_token} {dialog} {sys_token} "
)

CURRENT_STEP_TEMP = "We are on Step {step_num}: {step_text}"
NO_STEP_TEMP = "We are just starting the recipe"



CONTEXT_WINDOW = 3

# ======== DEFAULT ARGUMENTS ========
DEFAULT_EPOCHS = 3.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_WARMUP_STEPS = 100
DEFAULT_LOGGING_STEPS = 100
DEFAULT_EVAL_STEPS = 1000
DEFAULT_SAVE_INTERVAL = 5
DEFAULT_SAVE_LIMIT = 4
BLOCK_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-5
MAX_LEN = 512
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_GRAD_ACC_STEPS = 1
DEFAULT_TEMPERATURE = 1.0
DEFAULT_STOP_CRITERIA = -1.0
DEFAULT_SEED = 11731
DEFAULT_LR_STEP_SIZE = 250
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_MAX_LEN = 512
