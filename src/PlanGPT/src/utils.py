
from transformers import PreTrainedModel

from src.constants import VICUNA_PROMPT_DICT, CURRENT_STEP_TEMP, NO_STEP_TEMP


def write_array_to_file(arr, file_path):
    print('Writing %d sentences to %s' % (len(arr), file_path))
    with open(file_path, 'w', encoding='utf8') as out_file:
        for item in arr:
            out_file.write(str(item).encode('utf8').decode('utf8') + '\n')


def safe_save_model_for_hf_trainer(model: PreTrainedModel, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = model.state_dict()
    cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
    del state_dict
    model.save_pretrained(output_dir, state_dict=cpu_state_dict)


def clean_system_response(response):
    response = response.replace("<break time=\"200ms\"/>", "")
    response = response.replace("<break time=\"100ms\"/>", "")
    response = response.replace("<break time=\"300ms\"/>", "")
    response = response.replace("<amazon:emotion name=\"excited\" intensity=\"low\">", "")
    response = response.replace("<amazon:emotion name=\"excited\" intensity=\"medium\">", "")
    response = response.replace("<amazon:emotion name=\"excited\" intensity=\"high\">", "")
    response = response.replace("</amazon:emotion>", "")
    response = response.replace("     ", " ")
    response = response.replace("    ", " ")
    response = response.replace("   ", " ")
    response = response.replace("  ", " ")
    response = response.replace(" .", ".")
    return response.strip()


def is_conversational_model(model_type):
    return "openassitant" in model_type.lower() or "vicuna" in model_type.lower()


def build_vicuna_prompt(d):
    SYS_TOKEN = 'Assistant:'
    USER_TOKEN = 'Human:'
    SEP_TOKEN = '###'

    # NEW VICUNA TOKENS
    SYS_TOKEN = '<|assistant|>'
    USER_TOKEN = '<|prompter|>'
    SEP_TOKEN = '<|endoftext|>'

    return VICUNA_PROMPT_DICT.format(
        user_token=USER_TOKEN, sys_token=SYS_TOKEN, sep_token=SEP_TOKEN, recipe=d["recipe"],
        system_tone=d['system_tone'].replace('_', ' '),
        dialog=' '.join([f"{USER_TOKEN if t['speaker'] == 'user' else SYS_TOKEN} {t['text']} {SEP_TOKEN}" for t in d['dialog']]),
        current_step=CURRENT_STEP_TEMP.format(step_num=d['current_step']['num'], step_text=d['current_step']['text']) if d['current_step']['num'] > 0 else NO_STEP_TEMP)
