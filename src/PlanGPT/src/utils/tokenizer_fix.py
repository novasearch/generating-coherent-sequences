import transformers

TOKENIZER_PATH = "vicuna/vicuna_7b"

tokenizer = transformers.LlamaTokenizer.from_pretrained(
        TOKENIZER_PATH,
        cache_dir='.cache',
        model_max_length=512,
        padding_side="right",
        use_fast=False,
)

print(f"Tokenizer max length: {tokenizer.model_max_length}")

tokenizer.pad_token = tokenizer.unk_token

print(f"Tokenizer bos token: {tokenizer.bos_token}")
print(f"Tokenizer eos token: {tokenizer.eos_token}")
print(f"Tokenizer pad token: {tokenizer.pad_token}")
print(f"Tokenizer unk token: {tokenizer.unk_token}")

tokenizer.save_pretrained("vicuna/vicuna_7b_fixed")



