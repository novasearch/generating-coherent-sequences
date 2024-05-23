

IGNORE_INDEX = -100

CONVERSATIONAL_PROMPT_FORMAT = "<|prompter|> I will give you a recipe and I want you to help me do it step by step. Please use a {system_tone} tone of voice. Recipe: {recipe} This is the current step: {current_step}. <|endoftext|> <|assistant|> ok! <|endoftext|> {dialog} <|endoftext|> <|assistant|> "
