import torch
import torch.multiprocessing as mp
import json
import os
import sys
import time
from math import ceil
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from safetensors.torch import load_file

try:
    from model import BioQwenModel
except ImportError:
    print("Error: Could not import BioQwenModel from model.py.")
    sys.exit(1)

BATCH_SIZE = 32  
CHECKPOINT_FILE = "model/model.safetensors" # Please download the model from our anonymous Hugging Face
BASE_MODEL_PATH = "model/qwen3-8b-multimodal" # Please download the model from our anonymous Hugging Face

ESM_EMB_PATH = "../data/protein_embeddings.pt"
GNN1_EMB_PATH = "../data/Knowledge_Graph/pathway_subgraph_embeddings_one_hop_large_1024_verified_undirected.pth"
GNN2_EMB_PATH = "../data/Knowledge_Graph/gene_subgraph_embeddings_one_hop_large_1024_verified_undirected.pth"

JSON_DATA_PATHS = [
    r"../data/Test_Dataset_Augmented_Prompt/HepG2.json",
    r"../data/Test_Dataset_Augmented_Prompt/Jurkat.json",
    r"../data/Test_Dataset_Augmented_Prompt/K562.json",
    r"../data/Test_Dataset_Augmented_Prompt/RPE1.json"
]

OUTPUT_DIR = "../data/Knowledge_Graph/grpo_inference_without_rag_results"
TEMPLATE_PATH = os.path.join(BASE_MODEL_PATH, "chat_template.jinja")


def load_and_process_embedding(path, type_name):
    data = torch.load(path, map_location="cpu")
    if type_name == 'ESM':
        names = list(data.keys())
        tensors = [v.float() for v in data.values()]
        matrix = torch.stack(tensors)
    else:
        names = data['gene_names']
        matrix = data['embeddings'].float()

    mean_vec = matrix.mean(dim=0, keepdim=True)
    final_matrix = torch.cat([matrix, mean_vec], dim=0)
    name_to_id = {name: i for i, name in enumerate(names)}
    unknown_idx = len(names)
    return final_matrix, name_to_id, unknown_idx


class BioInference:
    def __init__(self, device_id=0):
        self.device = f"cuda:{device_id}"
        print(f"[GPU {device_id}] Initializing Inference Model...")

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if os.path.exists(TEMPLATE_PATH):
            with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
                self.tokenizer.chat_template = f.read()

        esm_matrix, self.esm_map, self.esm_unk = load_and_process_embedding(ESM_EMB_PATH, 'ESM')
        gnn1_matrix, self.gnn1_map, self.gnn1_unk = load_and_process_embedding(GNN1_EMB_PATH, 'GNN1')
        gnn2_matrix, self.gnn2_map, self.gnn2_unk = load_and_process_embedding(GNN2_EMB_PATH, 'GNN2')

        target_tokens = ["<ESM2_EMB>", "<GNN_EMB_1>", "<GNN_EMB_2>"]
        self.token_map = {t: self.tokenizer.convert_tokens_to_ids(t) for t in target_tokens}

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16, lora_alpha=32, lora_dropout=0.1, target_modules="all-linear"
        )
        base_model = get_peft_model(base_model, peft_config)

        self.model = BioQwenModel(
            base_llm=base_model,
            esm_matrix=esm_matrix,
            gnn1_matrix=gnn1_matrix,
            gnn2_matrix=gnn2_matrix,
            token_map=self.token_map
        )

        state_dict = load_file(CHECKPOINT_FILE)
        self.model.load_state_dict(state_dict, strict=False)

        self.model.to(device=self.device, dtype=torch.bfloat16)
        self.model.eval()

        self.llm = self.model.llm
        self.esm_interaction = self.model.esm_interaction
        self.gnn1_interaction = self.model.gnn1_interaction
        self.gnn2_interaction = self.model.gnn2_interaction

        self.esm_emb_layer = self.model.esm_emb_layer
        self.gnn1_emb_layer = self.model.gnn1_emb_layer
        self.gnn2_emb_layer = self.model.gnn2_emb_layer

        print(f"[GPU {device_id}] Ready.")

    def predict_batch(self, batch_data, max_new_tokens=512):
        prompts = []
        p_genes = []
        t_genes = []

        for item in batch_data:
            messages = [{"role": "user", "content": item['prompt']}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(text)
            p_genes.append(item['pert'])
            t_genes.append(item['gene'])

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        input_len = input_ids.shape[1]

        if hasattr(self.llm, "get_input_embeddings"):
            embed_layer = self.llm.get_input_embeddings()
        elif hasattr(self.llm, "base_model") and hasattr(self.llm.base_model, "model") and hasattr(self.llm.base_model.model, "get_input_embeddings"):
            embed_layer = self.llm.base_model.model.get_input_embeddings()
        else:
            embed_layer = self.llm.base_model.model.model.embed_tokens

        inputs_embeds = embed_layer(input_ids)

        def batch_compute_and_replace(map_dict, unk_idx, raw_emb_layer, interaction_mod, token_str):
            p_indices = [map_dict.get(g, unk_idx) for g in p_genes]
            t_indices = [map_dict.get(g, unk_idx) for g in t_genes]

            p_tensor = torch.tensor(p_indices, device=self.device)
            t_tensor = torch.tensor(t_indices, device=self.device)

            p_raw = raw_emb_layer(p_tensor).unsqueeze(1).to(dtype=torch.bfloat16)
            t_raw = raw_emb_layer(t_tensor).unsqueeze(1).to(dtype=torch.bfloat16)

            fused = interaction_mod(p_raw, t_raw)

            token_id = self.token_map[token_str]
            mask = (input_ids == token_id)

            if mask.any():
                try:
                    inputs_embeds[mask] = fused.view(-1, inputs_embeds.shape[-1])
                except RuntimeError as e:
                    print(f"Embedding replace error for {token_str}: {e}")

        batch_compute_and_replace(self.esm_map, self.esm_unk, self.esm_emb_layer, self.esm_interaction, "<ESM2_EMB>")
        batch_compute_and_replace(self.gnn1_map, self.gnn1_unk, self.gnn1_emb_layer, self.gnn1_interaction, "<GNN_EMB_1>")
        batch_compute_and_replace(self.gnn2_map, self.gnn2_unk, self.gnn2_emb_layer, self.gnn2_interaction, "<GNN_EMB_2>")

        with torch.no_grad():
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )

        real_outputs = []
        for out_seq in outputs:
            if out_seq.shape[0] < input_len:
                new_tokens = out_seq
            else:
                new_tokens = out_seq[input_len:]

            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            real_outputs.append(text)

        return real_outputs


def gpu_worker(rank, gpu_id, data_chunk, job_tag):
    wait_time = rank * 10
    print(f"Worker {rank} (GPU {gpu_id}, job={job_tag}) waiting {wait_time}s...")
    time.sleep(wait_time)

    try:
        predictor = BioInference(device_id=gpu_id)
    except Exception as e:
        print(f"Worker {rank} initialization failed: {e}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"result_{job_tag}_gpu_{gpu_id}.jsonl")

    existing_keys = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    existing_keys.add(f"{d['pert']}_{d['gene']}")
                except Exception:
                    pass

    todo_data = []
    for item in data_chunk:
        key = f"{item['pert']}_{item['gene']}"
        if key not in existing_keys:
            todo_data.append(item)

    print(f"Worker {rank} (GPU {gpu_id}, job={job_tag}) has {len(todo_data)} items to process (Batch Size: {BATCH_SIZE}).")

    batches = [todo_data[i:i + BATCH_SIZE] for i in range(0, len(todo_data), BATCH_SIZE)]

    with open(output_file, "a+", encoding="utf-8") as f_out:
        for batch in tqdm(batches, desc=f"GPU {gpu_id} [{job_tag}]", position=rank):
            try:
                results = predictor.predict_batch(batch, max_new_tokens=512)

                for item, res_text in zip(batch, results):
                    out_item = {
                        "pert": item['pert'],
                        "gene": item['gene'],
                        "gold_label": item['label'],
                        "model_output": res_text
                    }
                    f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")

                f_out.flush()

            except Exception as e:
                print(f"Error in batch on GPU {gpu_id}, job={job_tag}: {e}")
                for item in batch:
                    err_item = {
                        "pert": item['pert'],
                        "gene": item['gene'],
                        "error": str(e)
                    }
                    f_out.write(json.dumps(err_item, ensure_ascii=False) + "\n")
                f_out.flush()

    print(f"Worker {rank} (GPU {gpu_id}, job={job_tag}) Finished!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    TARGET_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
    NUM_GPUS = len(TARGET_GPUS)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for file_idx, json_path in enumerate(JSON_DATA_PATHS):
        job_tag = os.path.splitext(os.path.basename(json_path))[0]

        print(f"\n===== Processing file {file_idx + 1}/{len(JSON_DATA_PATHS)}: {json_path} =====")
        with open(json_path, "r", encoding="utf-8") as f:
            full_data = json.load(f)

        print(f"Total Samples in {job_tag}: {len(full_data)}")
        if len(full_data) == 0:
            print(f"Warning: {job_tag} has 0 samples, skipping.")
            continue

        chunk_size = ceil(len(full_data) / NUM_GPUS)
        chunks = [full_data[i:i + chunk_size] for i in range(0, len(full_data), chunk_size)]

        print(f"Launching {len(chunks)} processes on GPUs {TARGET_GPUS}. Batch Size = {BATCH_SIZE}")

        processes = []
        for rank, gpu_id in enumerate(TARGET_GPUS):
            if rank < len(chunks):
                p = mp.Process(target=gpu_worker, args=(rank, gpu_id, chunks[rank], job_tag))
                p.start()
                processes.append(p)

        for p in processes:
            p.join()

        print(f"\nMerging results for {job_tag}...")
        final_output = os.path.join(OUTPUT_DIR, f"inference_results_{job_tag}.jsonl")
        with open(final_output, 'w', encoding='utf-8') as f_out:
            for gpu_id in TARGET_GPUS:
                part_path = os.path.join(OUTPUT_DIR, f"result_{job_tag}_gpu_{gpu_id}.jsonl")
                if os.path.exists(part_path):
                    with open(part_path, 'r', encoding='utf-8') as f_in:
                        for line in f_in:
                            f_out.write(line)

        print(f"Done for {job_tag}! Results saved to {final_output}")

    print("\nAll files processed.")