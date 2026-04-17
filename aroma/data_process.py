import torch
from torch.utils.data import Dataset
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Any

class GenePerturbationDataset(Dataset):
    def __init__(self, json_path, tokenizer, 
                 esm_name_to_id, esm_unknown_idx,
                 gnn1_name_to_id, gnn1_unknown_idx, 
                 gnn2_name_to_id, gnn2_unknown_idx,
                 max_length=2048): 
        
        self.tokenizer = tokenizer
        self.max_len = max_length
        
        print(f"Loading data from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        print(f"Loaded {len(self.raw_data)} samples.")
        
        required_tokens = ["<ESM2_EMB>", "<GNN_EMB_1>", "<GNN_EMB_2>"]
        for i, item in enumerate(self.raw_data):
            text = item.get('instruction', "")
            if not all(t in text for t in required_tokens):
                raise ValueError(f"Error: Sample {i} is missing required Embedding Tokens. Please check your JSON file.")
        
        print("Data integrity check passed: All samples contain necessary tokens.")

        self.esm_map = esm_name_to_id
        self.esm_unk = esm_unknown_idx
        
        self.gnn1_map = gnn1_name_to_id
        self.gnn1_unk = gnn1_unknown_idx
        
        self.gnn2_map = gnn2_name_to_id
        self.gnn2_unk = gnn2_unknown_idx
        
        template_path = "model/chat_template.jinja"  # Please download the model from our anonymous Hugging Face
        
        print(f"Loading Chat Template from {template_path}...")
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                self.tokenizer.chat_template = f.read()
            print("Chat Template loaded and registered to Tokenizer.")
        else:
            raise FileNotFoundError(f"Template file not found: {template_path}")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]

        user_content = item['instruction']
        if item['input']:
            user_content += "\n" + item['input']

        messages = [
            {"role": "user", "content": user_content}
        ]
        
        full_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True 
        )

        prompt_text = full_prompt
        output_text = item['output'] + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=True, truncation=True, max_length=self.max_len
        )['input_ids']
        
        output_ids = self.tokenizer(
            output_text, add_special_tokens=False, truncation=True, max_length=self.max_len
        )['input_ids']

        input_ids = prompt_ids + output_ids
        labels = [-100] * len(prompt_ids) + output_ids

        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            labels = labels[:self.max_len]
        
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = input_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        target_pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        attention_mask = input_ids.ne(target_pad_id).long()

        p_gene = item.get('pert_gene', 'UNKNOWN') 
        t_gene = item.get('target_gene', 'UNKNOWN')

        p_esm_idx = self.esm_map.get(p_gene, self.esm_unk)
        t_esm_idx = self.esm_map.get(t_gene, self.esm_unk)

        p_gnn1_idx = self.gnn1_map.get(p_gene, self.gnn1_unk)
        t_gnn1_idx = self.gnn1_map.get(t_gene, self.gnn1_unk)

        p_gnn2_idx = self.gnn2_map.get(p_gene, self.gnn2_unk)
        t_gnn2_idx = self.gnn2_map.get(t_gene, self.gnn2_unk)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pert_esm_idx': torch.tensor(p_esm_idx, dtype=torch.long),
            'target_esm_idx': torch.tensor(t_esm_idx, dtype=torch.long),
            'pert_gnn1_idx': torch.tensor(p_gnn1_idx, dtype=torch.long),
            'target_gnn1_idx': torch.tensor(t_gnn1_idx, dtype=torch.long),
            'pert_gnn2_idx': torch.tensor(p_gnn2_idx, dtype=torch.long),
            'target_gnn2_idx': torch.tensor(t_gnn2_idx, dtype=torch.long),
        }

@dataclass
class MultiModalDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        for key in features[0].keys():
            batch[key] = torch.stack([f[key] for f in features])
        return batch