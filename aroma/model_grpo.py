import torch
import torch.nn as nn
from model import BioQwenModel 

class BioQwenModelForGRPO(BioQwenModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warnings_issued = {}

    def add_model_tags(self, tags):
        if not hasattr(self.config, "model_tags") or self.config.model_tags is None:
            self.config.model_tags = []
        if isinstance(tags, str): tags = [tags]
        for tag in tags:
            if tag not in self.config.model_tags:
                self.config.model_tags.append(tag)

    @property
    def is_gradient_checkpointing(self):
        return getattr(self.llm, "is_gradient_checkpointing", False)

    @is_gradient_checkpointing.setter
    def is_gradient_checkpointing(self, value):
        if hasattr(self.llm, "is_gradient_checkpointing"):
            self.llm.is_gradient_checkpointing = value

    @property
    def can_generate(self):
        return getattr(self.llm, "can_generate", True)

    def push_to_hub(self, *args, **kwargs):
        print("[BioQwenModelForGRPO] Warning: push_to_hub called but not implemented. Ignoring.")
        return None

    def get_input_embeddings(self):
        if hasattr(self.llm, "get_input_embeddings"):
            return self.llm.get_input_embeddings()
        elif hasattr(self.llm.model, "get_input_embeddings"):
            return self.llm.model.get_input_embeddings()
        return self.llm.model.model.embed_tokens
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        
        gradient_checkpointing_kwargs["use_reentrant"] = False
        
        if hasattr(self.llm, "base_model") and hasattr(self.llm.base_model, "gradient_checkpointing_enable"):
            self.llm.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        else:
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def forward(self, input_ids, attention_mask=None, labels=None, 
                pert_esm_idx=None, target_esm_idx=None, 
                pert_gnn1_idx=None, target_gnn1_idx=None,
                pert_gnn2_idx=None, target_gnn2_idx=None,
                **kwargs):
        
        embed_layer = self.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)
        
        if self.training and inputs_embeds.requires_grad is False:
            inputs_embeds.requires_grad_(True)
        
        inputs_embeds = inputs_embeds.clone()
        
        if pert_esm_idx is not None:
            target_dtype = inputs_embeds.dtype
            
            def process_stream(raw_emb_layer, interaction_mod, p_idx, t_idx):
                p_emb = raw_emb_layer(p_idx).unsqueeze(1).to(target_dtype)
                t_emb = raw_emb_layer(t_idx).unsqueeze(1).to(target_dtype)
                return interaction_mod(p_emb, t_emb)

            fused_esm = process_stream(self.esm_emb_layer, self.esm_interaction, pert_esm_idx, target_esm_idx)
            self._safe_replace(input_ids, inputs_embeds, '<ESM2_EMB>', fused_esm)
            
            fused_gnn1 = process_stream(self.gnn1_emb_layer, self.gnn1_interaction, pert_gnn1_idx, target_gnn1_idx)
            self._safe_replace(input_ids, inputs_embeds, '<GNN_EMB_1>', fused_gnn1)
            
            fused_gnn2 = process_stream(self.gnn2_emb_layer, self.gnn2_interaction, pert_gnn2_idx, target_gnn2_idx)
            self._safe_replace(input_ids, inputs_embeds, '<GNN_EMB_2>', fused_gnn2)

        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)

    def generate(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None and input_ids is not None:
            embed_layer = self.get_input_embeddings()
            inputs_embeds = embed_layer(input_ids)
        
        pert_esm_idx = kwargs.get('pert_esm_idx', None)
        target_esm_idx = kwargs.get('target_esm_idx', None)
        pert_gnn1_idx = kwargs.get('pert_gnn1_idx', None)
        target_gnn1_idx = kwargs.get('target_gnn1_idx', None)
        pert_gnn2_idx = kwargs.get('pert_gnn2_idx', None)
        target_gnn2_idx = kwargs.get('target_gnn2_idx', None)

        if pert_esm_idx is not None:
            target_dtype = inputs_embeds.dtype
            device = inputs_embeds.device

            def inject_embeddings(p_idx, t_idx, raw_emb_layer, interaction_mod, token_str):
                if p_idx is None or t_idx is None: return
                p_idx, t_idx = p_idx.to(device), t_idx.to(device)
                p_emb = raw_emb_layer(p_idx).unsqueeze(1).to(target_dtype)
                t_emb = raw_emb_layer(t_idx).unsqueeze(1).to(target_dtype)
                fused = interaction_mod(p_emb, t_emb) 
                
                if token_str in self.token_map:
                    token_id = self.token_map[token_str]
                    if input_ids is not None:
                        mask = (input_ids == token_id)
                        if mask.any():
                            try:
                                inputs_embeds[mask] = fused.view(-1, inputs_embeds.shape[-1])
                            except Exception as e:
                                if token_str not in self.warnings_issued:
                                    print(f"[Generation Warning] {token_str}: {e}")
                                    self.warnings_issued[token_str] = True

            inject_embeddings(pert_esm_idx, target_esm_idx, self.esm_emb_layer, self.esm_interaction, "<ESM2_EMB>")
            inject_embeddings(pert_gnn1_idx, target_gnn1_idx, self.gnn1_emb_layer, self.gnn1_interaction, "<GNN_EMB_1>")
            inject_embeddings(pert_gnn2_idx, target_gnn2_idx, self.gnn2_emb_layer, self.gnn2_interaction, "<GNN_EMB_2>")

        gen_kwargs = kwargs.copy()
        for k in ['pert_esm_idx', 'target_esm_idx', 'pert_gnn1_idx', 'target_gnn1_idx', 
                  'pert_gnn2_idx', 'target_gnn2_idx', 'pert_gene', 'target_gene', 'label', 'prompt']:
            gen_kwargs.pop(k, None)

        output_sequences = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs
        )
        
        if input_ids is not None:
            if output_sequences.device != input_ids.device:
                output_sequences = output_sequences.to(input_ids.device)
            return torch.cat([input_ids, output_sequences], dim=1)
            
        return output_sequences