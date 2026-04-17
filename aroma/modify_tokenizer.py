from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "model/qwen3-8b" # Please download the model from Hugging Face.
print(f"Loading tokenizer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)
print(f"Loading model from {model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto"
)
print(f"Original vocab size: {len(tokenizer)}")
print(f"Original embedding matrix shape: {model.get_input_embeddings().weight.shape}")

new_special_tokens = [
    "<Embedding_Start>", 
    "<Embedding_End>", 
    "<Embedding>"
]

num_added_toks = tokenizer.add_special_tokens(
    {'additional_special_tokens': new_special_tokens}
)

print(f"\nAdded {num_added_toks} new special tokens.")
print(f"New vocab size: {len(tokenizer)}")

print(f"\nModel resize skipped, preserving original matrix.")
print(f"Final model embedding matrix shape: {model.get_input_embeddings().weight.shape}")

print("\n--- Verification ---")
start_token = "<Embedding_Start>"
embed_token = "<Embedding>"

start_id = tokenizer.convert_tokens_to_ids(start_token)
embed_id = tokenizer.convert_tokens_to_ids(embed_token)

print(f"Token '{start_token}' -> ID: {start_id}")
print(f"Token '{embed_token}' -> ID: {embed_id}")

original_shape = model.get_input_embeddings().weight.shape
if start_id < original_shape[0] and embed_id < original_shape[0]:
    print(f"\n Success! New Token ID (e.g., {embed_id}) is within the original matrix range ({original_shape[0]}).")
    print("All original weights have been preserved.")
else:
    print(f"\n Failure! Token ID {embed_id} exceeds the original matrix range {original_shape[0]}.")

save_path = r"model/qwen3-8b-multimodal" 
print(f"\nSaving modified model AND tokenizer to {save_path}...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path) 
print(" Done.")