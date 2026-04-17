import torch
import torch.nn as nn
import torch.nn.functional as F

class BioInteractionModule(nn.Module):
    def __init__(self, input_dim, llm_dim, hidden_dim=1024): 
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, llm_dim)

    def forward(self, query_emb, key_emb):
        q = self.input_proj(query_emb)
        k = self.input_proj(key_emb)
        v = k 
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        out = self.norm(q + attn_out) 
        out = self.output_proj(out) 
        return out

class BioQwenModel(nn.Module):
    def __init__(self, base_llm, esm_matrix, gnn1_matrix, gnn2_matrix, token_map):
        super().__init__()
        self.llm = base_llm
        self.config = base_llm.config
        self.token_map = token_map
        self.llm_dim = base_llm.config.hidden_size
        
        self.esm_emb_layer = nn.Embedding.from_pretrained(esm_matrix, freeze=True)
        self.gnn1_emb_layer = nn.Embedding.from_pretrained(gnn1_matrix, freeze=True)
        self.gnn2_emb_layer = nn.Embedding.from_pretrained(gnn2_matrix, freeze=True)
        
        self.esm_interaction = BioInteractionModule(esm_matrix.shape[1], self.llm_dim)
        self.gnn1_interaction = BioInteractionModule(gnn1_matrix.shape[1], self.llm_dim)
        self.gnn2_interaction = BioInteractionModule(gnn2_matrix.shape[1], self.llm_dim)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def forward(self, input_ids, attention_mask=None, labels=None, 
                pert_esm_idx=None, target_esm_idx=None, 
                pert_gnn1_idx=None, target_gnn1_idx=None,
                pert_gnn2_idx=None, target_gnn2_idx=None,
                **kwargs):
        
        if hasattr(self.llm, "get_input_embeddings"):
            embed_layer = self.llm.get_input_embeddings()
        elif hasattr(self.llm.model, "get_input_embeddings"): 
             embed_layer = self.llm.model.get_input_embeddings()
        else:
            embed_layer = self.llm.model.model.embed_tokens 
        
        inputs_embeds = embed_layer(input_ids)
        
        if pert_esm_idx is not None:
            target_dtype = inputs_embeds.dtype

            def process_stream(emb_layer, interaction_mod, p_idx, t_idx):
                p_emb = emb_layer(p_idx).unsqueeze(1).to(target_dtype)
                t_emb = emb_layer(t_idx).unsqueeze(1).to(target_dtype)
                return interaction_mod(p_emb, t_emb)

            fused_esm = process_stream(self.esm_emb_layer, self.esm_interaction, pert_esm_idx, target_esm_idx)
            fused_gnn1 = process_stream(self.gnn1_emb_layer, self.gnn1_interaction, pert_gnn1_idx, target_gnn1_idx)
            fused_gnn2 = process_stream(self.gnn2_emb_layer, self.gnn2_interaction, pert_gnn2_idx, target_gnn2_idx)

            self._safe_replace(input_ids, inputs_embeds, '<ESM2_EMB>', fused_esm)
            self._safe_replace(input_ids, inputs_embeds, '<GNN_EMB_1>', fused_gnn1)
            self._safe_replace(input_ids, inputs_embeds, '<GNN_EMB_2>', fused_gnn2)

        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)

    def _safe_replace(self, input_ids, inputs_embeds, token_str, source_emb):
        token_id = self.token_map[token_str]
        mask = (input_ids == token_id) 
        if not mask.any():
            return
        indices = torch.nonzero(mask) 
        batch_indices = indices[:, 0] 
        seq_indices = indices[:, 1]   
        target_embeddings = source_emb[batch_indices, 0, :] 
        inputs_embeds[batch_indices, seq_indices] = target_embeddings.to(inputs_embeds.dtype)