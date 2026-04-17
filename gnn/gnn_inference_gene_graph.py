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

PATH_TO_WEIGHTS = os.path.join(BASE_DIR, f"gat_encoder_gene_graph_pretrained{FILE_SUFFIX}.pth")
PATH_TO_DATA = os.path.join(BASE_DIR, f"gene_graph_for_inference{FILE_SUFFIX}.pth")
PATH_TO_MAPPING = os.path.join(BASE_DIR, "gene_subgraph_names_mapping.pkl")

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
            
            if (local_idx + 1) % 500 == 0:
                print(f"Worker {rank} on cuda:{rank}: Processed {local_idx + 1}/{len(node_indices)} nodes.")

        final_embeddings_chunk = torch.stack(all_embeddings_chunk, dim=0)
        
        output_queue.put({
            'start_idx': node_indices[0],
            'embeddings': final_embeddings_chunk
        })
        print(f"Worker {rank} on cuda:{rank} finished.")

    except Exception as e:
        print(f"Worker {rank} encountered an error: {e}")
        output_queue.put({'start_idx': -1, 'embeddings': None})

def run_subgraph_inference_multigpu(data_path, weights_path, mapping_path, num_hops=1, num_gpus=8) -> torch.Tensor:
    try:
        print(f"Loading data: {data_path}")
        data_cpu = torch.load(data_path) 
        if not isinstance(data_cpu, Data):
            if isinstance(data_cpu, nx.Graph):
                 data_cpu = from_networkx(data_cpu)
            else:
                 print(f"Error: Loaded object is not a PyG Data or NetworkX graph.")
                 return
    except Exception as e:
        print(f"Error: Unable to load graph file {data_path}. Error: {e}")
        return

    try:
        with open(mapping_path, 'rb') as f:
            gene_names: List[str] = pickle.load(f)
        print(f"Successfully loaded {len(gene_names)} gene name mappings.")
    except FileNotFoundError:
        print(f"Error: Gene name mapping file not found: {mapping_path}")
        return
    except Exception as e:
        print(f"Error: Failed to load gene name mapping file: {e}")
        return
    
    if not hasattr(data_cpu, 'x') or data_cpu.x is None:
        print(f"Error: PyG Data object is missing node features 'x'.")
        return
        
    data_cpu.x = data_cpu.x.float()
    
    INITIAL_FEATURES_DIM = data_cpu.x.shape[1] 
    num_nodes = data_cpu.num_nodes
    
    if num_nodes != len(gene_names):
        print(f"Warning: Node count ({num_nodes}) does not match loaded gene names count ({len(gene_names)})!")
        return

    GNN_HIDDEN_CHANNELS = 64   
    GNN_OUTPUT_CHANNELS = 1024 
    
    model_config = {
        'in_channels': INITIAL_FEATURES_DIM, 
        'hidden_channels': GNN_HIDDEN_CHANNELS,
        'out_channels': GNN_OUTPUT_CHANNELS,
        'num_layers': 3, 
        'heads': 4
    }
    
    print(f"Model Configuration: {model_config}")

    node_indices = list(range(num_nodes))
    chunks = []
    chunk_size = num_nodes // num_gpus
    remainder = num_nodes % num_gpus
    
    current_idx = 0
    for i in range(num_gpus):
        size = chunk_size + (1 if i < remainder else 0)
        chunks.append(node_indices[current_idx:current_idx + size])
        current_idx += size
    
    print(f"Total nodes: {num_nodes}. Allocating to {num_gpus} GPUs for inference (Hops={num_hops}).")
    
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

    collected_results = []
    for _ in range(num_gpus):
        collected_results.append(output_queue.get())
        
    for p in processes:
        p.join()
        
    print("All processes completed, starting result aggregation...")

    final_embeddings = torch.zeros(num_nodes, GNN_OUTPUT_CHANNELS)
    
    for res in collected_results:
        if res['start_idx'] == -1 or res['embeddings'] is None:
            print("Aggregation failed: One or more GPU processes encountered an error.")
            return
            
        start_idx = res['start_idx']
        embeddings_chunk = res['embeddings']
        
        end_idx = start_idx + embeddings_chunk.size(0)
        final_embeddings[start_idx:end_idx, :] = embeddings_chunk
    
    save_dir = os.path.dirname(data_path)
    file_name = f"gene_subgraph_embeddings_one_hop{FILE_SUFFIX}.pth"
    save_path_embed = os.path.join(save_dir, file_name)
    
    output_data = {
        "gene_names": gene_names,       
        "embeddings": final_embeddings  
    }
    torch.save(output_data, save_path_embed) 

    print(f"\nSubgraph-level embeddings generation completed.")
    print(f"Contains names and data for {len(gene_names)} genes.")
    print(f"Result shape: {final_embeddings.shape}")
    print(f"Saved to: {save_path_embed}")

    print("\n--- Example Results ---")
    for i in range(min(num_nodes, 5)): 
        gene_name = gene_names[i]
        embedding_vector = final_embeddings[i]
        embed_repr = embedding_vector[:5].numpy() 
        print(f"Gene: {gene_name} | Embedding (first 5 dimensions): {embed_repr}")
        
    print(f"--------------------------------------------------")
    
    return final_embeddings

if __name__ == '__main__':
    print(f"Weights file used: {PATH_TO_WEIGHTS}")
    print(f"Data file used: {PATH_TO_DATA}")
    print(f"Mapping file used: {PATH_TO_MAPPING}")
    
    if not os.path.exists(PATH_TO_DATA):
        print(f"Data file not found: {PATH_TO_DATA}")
    elif not os.path.exists(PATH_TO_WEIGHTS):
        print(f"Weights file not found: {PATH_TO_WEIGHTS}")
    elif not os.path.exists(PATH_TO_MAPPING):
        print(f"Mapping file not found: {PATH_TO_MAPPING}")
    else:
        run_subgraph_inference_multigpu(
            PATH_TO_DATA, 
            PATH_TO_WEIGHTS, 
            PATH_TO_MAPPING, 
            num_hops=1, 
            num_gpus=8
        )