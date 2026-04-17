import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import pickle
import torch.multiprocessing as mp
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import k_hop_subgraph, from_networkx
from torch_geometric.data import Data 
from typing import List

FILE_SUFFIX = "_large_1024_verified_undirected"
BASE_DIR = "../data/Knowledge_Graph"

PATH_TO_WEIGHTS = os.path.join(BASE_DIR, f"gat_encoder_path_graph_pretrained{FILE_SUFFIX}.pth")
PATH_TO_DATA = os.path.join(BASE_DIR, f"pathway_graph_for_inference{FILE_SUFFIX}.pth")
PATH_TO_MAPPING = os.path.join(BASE_DIR, "pathway_subgraph_names_mapping.pkl")

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=4):
        super(GATEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False))
        self.dropout = 0.5 

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        return x 

@torch.no_grad()
def generate_subgraph_embedding(data: Data, gnn_encoder: GATEncoder, node_idx: int, num_hops: int, device: torch.device):
    gnn_encoder.eval()
    
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=torch.tensor([node_idx]).to(device),
        num_hops=num_hops,
        edge_index=data.edge_index, 
        num_nodes=data.num_nodes,
        relabel_nodes=True 
    )
    
    if subset.numel() == 0:
        return torch.zeros(gnn_encoder.out_channels).to(device)

    sub_x = data.x[subset] 
    sub_z = gnn_encoder(sub_x, sub_edge_index)
    
    batch = torch.zeros(sub_z.size(0), dtype=torch.long).to(device)
    graph_embedding = global_mean_pool(sub_z, batch)
    
    return graph_embedding.squeeze() 

def worker_inference(rank, gnn_encoder_class, model_config, weights_path, num_hops, data_cpu, node_indices, output_queue):
    device = torch.device(f'cuda:{rank}')
    try:
        model = gnn_encoder_class(**model_config).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        worker_data = data_cpu.to(device) 
        all_embeddings_chunk = []
        
        for local_idx, global_node_idx in enumerate(node_indices):
            embed = generate_subgraph_embedding(
                data=worker_data, 
                gnn_encoder=model, 
                node_idx=global_node_idx, 
                num_hops=num_hops, 
                device=device
            )
            all_embeddings_chunk.append(embed.cpu()) 
            
            if (local_idx + 1) % 1000 == 0:
                print(f"[Worker {rank}] Processed {local_idx + 1}/{len(node_indices)}")

        final_embeddings_chunk = torch.stack(all_embeddings_chunk, dim=0)
        output_queue.put({'start_idx': node_indices[0], 'embeddings': final_embeddings_chunk})
        print(f"[Worker {rank}] Finished.")

    except Exception as e:
        print(f"[Worker {rank}] Error: {e}")
        output_queue.put({'start_idx': -1, 'embeddings': None})

def run_subgraph_inference_multigpu(data_path, weights_path, mapping_path, num_hops=1, num_gpus=8):
    print(f"Loading data: {data_path}")
    try:
        data_cpu = torch.load(data_path) 
        if isinstance(data_cpu, nx.Graph): data_cpu = from_networkx(data_cpu)
    except Exception as e:
        print(f"Error: Unable to load graph data: {e}"); return

    print(f"Loading mapping: {mapping_path}")
    try:
        with open(mapping_path, 'rb') as f:
            gene_names = pickle.load(f)
    except Exception as e:
        print(f"Error: Unable to load mapping file: {e}"); return

    if data_cpu.num_nodes != len(gene_names):
        print(f"Error: Graph node count ({data_cpu.num_nodes}) does not match mapping list length ({len(gene_names)})")
        return

    GNN_HIDDEN_CHANNELS = 64  
    GNN_OUTPUT_CHANNELS = 1024 
    
    if data_cpu.x is None: 
        print("Error: Data is missing features x")
        return
        
    data_cpu.x = data_cpu.x.float()
    INITIAL_FEATURES_DIM = data_cpu.x.shape[1] 

    model_config = {
        'in_channels': INITIAL_FEATURES_DIM, 
        'hidden_channels': GNN_HIDDEN_CHANNELS,
        'out_channels': GNN_OUTPUT_CHANNELS,
        'num_layers': 3, 
        'heads': 4
    }

    num_nodes = data_cpu.num_nodes
    node_indices = list(range(num_nodes))
    chunks = [node_indices[i::num_gpus] for i in range(num_gpus)] 

    print(f"Preparing inference for {num_nodes} nodes on {num_gpus} GPUs with {num_hops}-hop subgraph extraction...")
    
    mp.set_start_method('spawn', force=True) 
    output_queue = mp.Queue()
    processes = []
    
    for rank in range(num_gpus):
        p = mp.Process(
            target=worker_inference,
            args=(rank, GATEncoder, model_config, weights_path, num_hops, data_cpu, chunks[rank], output_queue)
        )
        processes.append(p)
        p.start()

    results_dict = {} 
    for _ in range(num_gpus):
        res = output_queue.get()
        if res['start_idx'] != -1:
            results_dict[res['start_idx']] = res['embeddings']
        else:
            print("Error: A worker failed during the inference process.")
            return

    for p in processes: p.join()

    final_embeddings = torch.zeros(num_nodes, GNN_OUTPUT_CHANNELS)
    
    for rank in range(num_gpus):
        start_key = chunks[rank][0] 
        if start_key in results_dict:
            indices = chunks[rank]
            data_chunk = results_dict[start_key]
            final_embeddings[indices] = data_chunk
        else:
            print(f"Error: Missing data chunk for rank {rank}")

    save_path_embed = os.path.join(BASE_DIR, f"pathway_subgraph_embeddings_one_hop{FILE_SUFFIX}.pth")
    
    output_data = {
        "gene_names": gene_names,
        "embeddings": final_embeddings
    }
    torch.save(output_data, save_path_embed)

    print(f"\nInference completed. Results saved to: {save_path_embed}")
    print(f"Total genes processed: {len(gene_names)}")
    print(f"Embedding shape: {final_embeddings.shape}")
    
    print("\n--- Examples ---")
    for i in range(min(5, num_nodes)):
        print(f"{gene_names[i]}: {final_embeddings[i][:4].tolist()}...")

if __name__ == '__main__':
    if not os.path.exists(PATH_TO_DATA):
        print(f"Error: Data file not found: {PATH_TO_DATA}")
    elif not os.path.exists(PATH_TO_WEIGHTS):
        print(f"Error: Weights file not found: {PATH_TO_WEIGHTS}")
    elif not os.path.exists(PATH_TO_MAPPING):
        print(f"Error: Mapping file not found: {PATH_TO_MAPPING}")
    else:
        run_subgraph_inference_multigpu(PATH_TO_DATA, PATH_TO_WEIGHTS, PATH_TO_MAPPING, num_hops=1, num_gpus=8)