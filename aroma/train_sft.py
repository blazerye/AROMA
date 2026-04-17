import os
import torch
import pickle
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from data_process import GenePerturbationDataset, MultiModalDataCollator 
from model import BioQwenModel 
from transformers import TrainerCallback 

class SaveMultimodalCallback(TrainerCallback):
    def __init__(self, model):
        self.model = model

    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.exists(checkpoint_folder):
            checkpoint_folder = args.output_dir
            
        save_path = os.path.join(checkpoint_folder, "multimodal_adapters.pth")
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        multimodal_components = {
            'esm_interaction': model_to_save.esm_interaction.state_dict(),
            'gnn1_interaction': model_to_save.gnn1_interaction.state_dict(),
            'gnn2_interaction': model_to_save.gnn2_interaction.state_dict(),
        }
        
        torch.save(multimodal_components, save_path)
        print(f"\n[Callback] Multimodal adapters saved to {save_path}")

def train():
    MODEL_ID = "model/qwen3-8b-multimodal" # Please download the model from Hugging Face.
    DATA_PATH = "../data/AROMA-Perturb-490k-Sample_1k_dataset.json" # The specific data will be open-sourced after the paper is accepted.
    OUTPUT_DIR = "../data/Knowledge_Graph/checkpoints_one_hop_1024_dimension_experiments"
    
    ESM_EMB_PATH = "../data/protein_embeddings.pt"
    GNN1_EMB_PATH = "../data/Knowledge_Graph/pathway_subgraph_embeddings_one_hop_large_1024_verified_undirected.pth"
    GNN2_EMB_PATH = "../data/Knowledge_Graph/gene_subgraph_embeddings_one_hop_large_1024_verified_undirected.pth"

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    target_tokens = ["<ESM2_EMB>", "<GNN_EMB_1>", "<GNN_EMB_2>"]
    token_map = {t: tokenizer.convert_tokens_to_ids(t) for t in target_tokens}
    print(f"Token check passed: {token_map}")

    def load_and_process_embedding(path, type_name):
        print(f"Processing {type_name} Embeddings from {path}...")
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
        
        print(f"{type_name} Matrix Ready. Shape: {final_matrix.shape}. Unknown Index: {unknown_idx}")
        return final_matrix, name_to_id, unknown_idx

    esm_matrix, esm_map, esm_unk = load_and_process_embedding(ESM_EMB_PATH, 'ESM')
    gnn1_matrix, gnn1_map, gnn1_unk = load_and_process_embedding(GNN1_EMB_PATH, 'GNN1')
    gnn2_matrix, gnn2_map, gnn2_unk = load_and_process_embedding(GNN2_EMB_PATH, 'GNN2')

    dataset = GenePerturbationDataset(
        json_path=DATA_PATH, 
        tokenizer=tokenizer, 
        esm_name_to_id=esm_map, esm_unknown_idx=esm_unk,
        gnn1_name_to_id=gnn1_map, gnn1_unknown_idx=gnn1_unk,
        gnn2_name_to_id=gnn2_map, gnn2_unknown_idx=gnn2_unk
    )
    collator = MultiModalDataCollator()

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules="all-linear"
    )

    base_model = get_peft_model(base_model, peft_config)

    final_model = BioQwenModel(
        base_llm=base_model,
        esm_matrix=esm_matrix,
        gnn1_matrix=gnn1_matrix,
        gnn2_matrix=gnn2_matrix,
        token_map=token_map
    )

    final_model.esm_emb_layer.weight.requires_grad = False
    final_model.gnn1_emb_layer.weight.requires_grad = False
    final_model.gnn2_emb_layer.weight.requires_grad = False

    trainable_modules = [
        final_model.gnn1_interaction, 
        final_model.gnn2_interaction, 
        final_model.esm_interaction
    ]
    
    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=1000,
        save_total_limit=20,
        remove_unused_columns=False,
        report_to="wandb",
        run_name="PertReason_multimodal",
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        deepspeed="../ds_config.json"
    )

    save_callback = SaveMultimodalCallback(final_model)

    trainer = Trainer(
        model=final_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[save_callback] 
    )

    def print_detailed_trainable_info(model):
        print("\n" + "="*50)
        print("MODEL PARAMETER INSPECTION")
        print("="*50)
        trainable_params = 0
        all_param = 0
        trainable_module_names = set()
        for name, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                prefix = name.split('.')[1] if hasattr(model, 'module') else name.split('.')[0]
                trainable_module_names.add(prefix)

        print(f"Total Parameters:     {all_param:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Trainable Ratio:      {100 * trainable_params / all_param:.4f}%")
        print("\n[ Active Training Modules ]")
        for name in sorted(list(trainable_module_names)):
            print(f"  - {name} ...")
        print("="*50 + "\n")

    print("Starting Training...")
    if training_args.local_rank in [-1, 0]:
        print_detailed_trainable_info(final_model)
        
    trainer.train()
    
    print("Training Finished. Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    final_save_path = os.path.join(OUTPUT_DIR, "multimodal_adapters.pth")
    multimodal_components = {
        'esm_interaction': final_model.esm_interaction.state_dict(),
        'gnn1_interaction': final_model.gnn1_interaction.state_dict(),
        'gnn2_interaction': final_model.gnn2_interaction.state_dict(),
    }
    torch.save(multimodal_components, final_save_path)
    print(f"Final Multimodal adapters saved to {final_save_path}")

if __name__ == "__main__":
    train()