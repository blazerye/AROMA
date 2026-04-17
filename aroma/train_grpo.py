import os
import re
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from safetensors.torch import load_file
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset as HFDataset
from trl.trainer.utils import selective_log_softmax

from data_process import GenePerturbationDataset 
from model_grpo import BioQwenModelForGRPO

BASE_MODEL_PATH = "model/qwen3-8b-multimodal" # Please download the model from our anonymous Hugging Face
SFT_CHECKPOINT_FILE = "../data/Knowledge_Graph/checkpoints_one_hop_1024_dimension_experiments"
DATA_PATH = "../data/grpo_multimodal_data.json"  # The specific data will be open-sourced after the paper is accepted.
ESM_EMB_PATH = "../data/protein_embeddings.pt"
GNN1_EMB_PATH = "../data/Knowledge_Graph/pathway_subgraph_embeddings_one_hop_large_1024_verified_undirected.pth"
GNN2_EMB_PATH = "../data/Knowledge_Graph/gene_subgraph_embeddings_one_hop_large_1024_verified_undirected.pth"
OUTPUT_DIR = "../data/Knowledge_Graph/grpo_checkpoints"

def load_and_process_embedding(path, type_name):
    data = torch.load(path, map_location="cpu")
    if type_name == 'ESM':
        names = list(data.keys())
        matrix = torch.stack([v.float() for v in data.values()])
    else:
        names = data['gene_names']
        matrix = data['embeddings'].float()
    
    mean_vec = matrix.mean(dim=0, keepdim=True)
    final_matrix = torch.cat([matrix, mean_vec], dim=0)
    name_to_id = {name: i for i, name in enumerate(names)}
    unknown_idx = len(names)
    return final_matrix, name_to_id, unknown_idx

def extract_xml_answer(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>")[-1].split("</answer>")[0].strip()
    return ""

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<(think|reasoning)>\s*[\s\S]*?\s*</(think|reasoning)>\s*<answer>\s*[\s\S]*?\s*</answer>\s*$"
    responses = []
    for c in completions:
        if isinstance(c, list) and isinstance(c[0], dict) and "content" in c[0]:
            responses.append(c[0]["content"])
        elif isinstance(c, dict) and "content" in c:
            responses.append(c["content"])
        else:
            responses.append(c)
    matches = [re.match(pattern, r.strip(), re.DOTALL) for r in responses]
    reward = [0.5 if m else 0.0 for m in matches]
    return reward

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = []
    for c in completions:
        if isinstance(c, list) and isinstance(c[0], dict) and "content" in c[0]:
            responses.append(c[0]["content"])
        elif isinstance(c, dict) and "content" in c:
            responses.append(c["content"])
        else:
            responses.append(c)
    extracted_responses = [extract_xml_answer(r) for r in responses]
    valid_answers = {"not significantly changed", "upregulated", "downregulated"}
    rewards = [0.5 if r.strip().lower() in valid_answers else 0.0 for r in extracted_responses]
    return rewards

def correctness_reward_func(prompts, completions, label, **kwargs) -> list[float]:
    responses = []
    for c in completions:
        if isinstance(c, list) and isinstance(c[0], dict) and "content" in c[0]:
            responses.append(c[0]["content"])
        elif isinstance(c, dict) and "content" in c:
            responses.append(c["content"])
        else:
            responses.append(c) 

    extracted_responses = [extract_xml_answer(r) for r in responses]
    extracted_answers = [extract_xml_answer(a) for a in label]

    rewards = []
    for r, a in zip(extracted_responses, extracted_answers):
        r, a = r.strip(), a.strip()
        if r == a:
            rewards.append(5.0)
        else:
            rewards.append(-1.0)
    return rewards

class BioGRPOTrainer(GRPOTrainer):
    MULTIMODAL_KEYS = [
        'pert_esm_idx', 'target_esm_idx', 
        'pert_gnn1_idx', 'target_gnn1_idx', 
        'pert_gnn2_idx', 'target_gnn2_idx'
    ]

    def _generate_and_score_completions(self, inputs):
        mm_kwargs_B = {}
        current_device = next(self.model.parameters()).device
        for k in self.MULTIMODAL_KEYS:
            if k in inputs[0]:
                raw_vals = [x[k] for x in inputs]
                if isinstance(raw_vals[0], torch.Tensor):
                    mm_kwargs_B[k] = torch.stack(raw_vals).to(current_device)
                else:
                    mm_kwargs_B[k] = torch.tensor(raw_vals, device=current_device)
        
        results = super()._generate_and_score_completions(inputs)
        mm_kwargs_BG = {}
        for k, v in mm_kwargs_B.items():
            mm_kwargs_BG[k] = v.repeat_interleave(self.num_generations, dim=0)
        results.update(mm_kwargs_BG)
        return results

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs: raise ValueError("Not supported")
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        mm_kwargs = {}
        for k in self.MULTIMODAL_KEYS:
            if k in inputs:
                mm_kwargs[k] = inputs[k]

        outputs = model(input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1, **mm_kwargs)
        logits = outputs.logits
        completion_logits = logits[:, -(logits_to_keep + 1) : -1, :]
        completion_ids = input_ids[:, -logits_to_keep:]
        per_token_logps = selective_log_softmax(completion_logits, completion_ids)
        
        per_token_kl = 0.0
        if self.beta != 0.0 and "ref_per_token_logps" in inputs:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        
        if self.beta != 0.0 and isinstance(per_token_kl, torch.Tensor):
            per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        else:
            per_token_loss = -per_token_loss

        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        return loss

def train():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    esm_m, esm_map, esm_unk = load_and_process_embedding(ESM_EMB_PATH, 'ESM')
    gnn1_m, gnn1_map, gnn1_unk = load_and_process_embedding(GNN1_EMB_PATH, 'GNN1')
    gnn2_m, gnn2_map, gnn2_unk = load_and_process_embedding(GNN2_EMB_PATH, 'GNN2')
    token_map = {t: tokenizer.convert_tokens_to_ids(t) for t in ["<ESM2_EMB>", "<GNN_EMB_1>", "<GNN_EMB_2>"]}

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules="all-linear"
    )
    base_model = get_peft_model(base_model, peft_config)
    
    model = BioQwenModelForGRPO(
        base_llm=base_model,
        esm_matrix=esm_m, gnn1_matrix=gnn1_m, gnn2_matrix=gnn2_m,
        token_map=token_map
    )
    
    state_dict = load_file(SFT_CHECKPOINT_FILE)
    model.load_state_dict(state_dict, strict=False)
    
    for param in model.esm_interaction.parameters(): param.requires_grad = False
    for param in model.gnn1_interaction.parameters(): param.requires_grad = False
    for param in model.gnn2_interaction.parameters(): param.requires_grad = False

    for name, param in model.named_parameters():
        if "lora" in name: param.requires_grad = True

    raw_dataset = GenePerturbationDataset(
        json_path=DATA_PATH, tokenizer=tokenizer,
        esm_name_to_id=esm_map, esm_unknown_idx=esm_unk,
        gnn1_name_to_id=gnn1_map, gnn1_unknown_idx=gnn1_unk,
        gnn2_name_to_id=gnn2_map, gnn2_unknown_idx=gnn2_unk
    )
    
    def generator():
        for i in range(len(raw_dataset)):
            item = raw_dataset.raw_data[i]
            processed = raw_dataset[i]
            user_content = item['instruction']
            if item['input']: user_content += "\n" + item['input']
            messages = [{"role": "user", "content": user_content}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            yield {
                "prompt": prompt_text,
                "label": item['output'],
                **{k: processed[k] for k in BioGRPOTrainer.MULTIMODAL_KEYS}
            }

    hf_dataset = HFDataset.from_generator(generator)
    split_dataset = hf_dataset.train_test_split(test_size=0.05)
    
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name="BioGRPO_Multimodal_Final_Fix",
        learning_rate=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_generations=16,
        max_prompt_length=2048,
        max_completion_length=2048,
        num_train_epochs=2,
        save_steps=50,
        max_grad_norm=1.0,
        report_to="wandb",
        temperature=1.2,
        top_p=0.9,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )

    trainer = BioGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[strict_format_reward_func, correctness_reward_func, int_reward_func],
        args=training_args,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test'],
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    
    torch.save({
        'esm_interaction': model.esm_interaction.state_dict(),
        'gnn1_interaction': model.gnn1_interaction.state_dict(),
        'gnn2_interaction': model.gnn2_interaction.state_dict(),
    }, os.path.join(OUTPUT_DIR, "multimodal_adapters_rl.pth"))

if __name__ == "__main__":
    train()