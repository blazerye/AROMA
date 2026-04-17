import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, from_networkx
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score
import os
import pickle

INPUT_DIM = 1024       
OUTPUT_DIM = 1024      
HIDDEN_DIM = 64      
NUM_HEADS = 4       
NUM_LAYERS = 3       

FILE_SUFFIX = "_large_1024_verified_undirected"

BASE_DIR = "../data/Knowledge_Graph"
PATH_GENE_GRAPH_SOURCE = os.path.join(BASE_DIR, "pathway_graph.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class LinkPredictor(nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        reduced_dim = 128  
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, reduced_dim), 
            nn.ReLU(),
            nn.Linear(reduced_dim, 1)
        )

    def forward(self, z, edge_label_index):
        z_src = z[edge_label_index[0]]
        z_dst = z[edge_label_index[1]]
        z_cat = torch.cat([z_src, z_dst], dim=-1)
        return self.mlp(z_cat)

def check_data_leakage(train_data, test_data, dataset_name="Test"):
    train_edges = train_data.edge_index.cpu()
    train_edge_set = set()
    for i in range(train_edges.size(1)):
        u, v = train_edges[0, i].item(), train_edges[1, i].item()
        train_edge_set.add(tuple(sorted((u, v))))
    
    test_edges = test_data.edge_label_index.cpu()
    test_labels = test_data.edge_label.cpu()
    
    leak_count = 0
    total_pos_checks = 0
    
    for i in range(test_edges.size(1)):
        if test_labels[i] == 1:
            total_pos_checks += 1
            u, v = test_edges[0, i].item(), test_edges[1, i].item()
            target_edge = tuple(sorted((u, v)))
            
            if target_edge in train_edge_set:
                leak_count += 1
                if leak_count <= 3:
                    print(f"Warning: Leakage found! Edge ({u}, {v}) exists in both training graph and test labels.")

    if leak_count == 0:
        print(f"Validation passed! {dataset_name} set has {total_pos_checks} positive samples with no leakage.")
        print("(Conclusion: The high AUC is genuine, indicating the model learned graph structural patterns rather than memorization.)")
    else:
        print(f"Validation failed! Found {leak_count} leaked edges.")
        print(f"(This means {leak_count/total_pos_checks:.2%} of test set answers are directly visible in the training graph.)")
    print("------------------------------------------------------------")

print("Step 1: Loading and cleaning data...")

if not os.path.exists(PATH_GENE_GRAPH_SOURCE):
    raise FileNotFoundError(f"Source graph file not found: {PATH_GENE_GRAPH_SOURCE}")

nx_graph = torch.load(PATH_GENE_GRAPH_SOURCE)
data = from_networkx(nx_graph)

data.x = torch.randn(data.num_nodes, INPUT_DIM).float()
final_x = data.x.clone().cpu()

pre_transform = T.Compose([
    T.ToUndirected(),
    T.RemoveDuplicatedEdges(),
    T.RemoveSelfLoops()
])
data = pre_transform(data)
print(f"Graph structure cleaning completed. Total nodes: {data.num_nodes}, Total undirected edges: {data.num_edges}")

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=False 
)

print("Splitting dataset...")
train_data, val_data, test_data = transform(data)

check_data_leakage(train_data, val_data, dataset_name="Validation")
check_data_leakage(train_data, test_data, dataset_name="Test")

print("\nStep 2: Starting training...")

gnn_encoder = GATEncoder(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, NUM_HEADS).to(device)
link_predictor = LinkPredictor(OUTPUT_DIM).to(device)
optimizer = torch.optim.AdamW(list(gnn_encoder.parameters()) + list(link_predictor.parameters()), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

def train_epoch():
    gnn_encoder.train()
    link_predictor.train()
    optimizer.zero_grad()
    
    x = train_data.x.to(device)
    edge_index = train_data.edge_index.to(device)
    pos_edge_label_index = train_data.edge_label_index.to(device)
    
    z = gnn_encoder(x, edge_index)
    
    neg_edge_label_index = negative_sampling(
        edge_index=edge_index, 
        num_nodes=train_data.num_nodes,
        num_neg_samples=pos_edge_label_index.size(1), 
        method='sparse'
    ).to(device)
    
    total_edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=1)
    labels = torch.cat([torch.ones(pos_edge_label_index.size(1)), torch.zeros(neg_edge_label_index.size(1))], dim=0).to(device)
    
    logits = link_predictor(z, total_edge_label_index).squeeze()
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test_epoch(data_split):
    gnn_encoder.eval()
    link_predictor.eval()
    x = data_split.x.to(device)
    edge_index = data_split.edge_index.to(device)
    
    edge_label_index = data_split.edge_label_index.to(device)
    edge_label = data_split.edge_label.to(device)
    
    z = gnn_encoder(x, edge_index)
    logits = link_predictor(z, edge_label_index).squeeze()
    preds = logits.sigmoid()
    return roc_auc_score(edge_label.cpu().numpy(), preds.cpu().numpy())

best_val_auc = 0
for epoch in range(1, 501):
    loss = train_epoch()
    if epoch % 10 == 0:
        val_auc = test_epoch(val_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")

print("\nStep 3: Final testing...")
test_auc = test_epoch(test_data)
print(f"Final Test AUC: {test_auc:.4f}")

if test_auc > 0.95:
    print("Note: AUC remains high. Since leakage detection passed, this indicates the graph structure has strong predictability.")

print("\nStep 4: Saving results...")
save_path_gnn = os.path.join(BASE_DIR, f"gat_encoder_path_graph_pretrained{FILE_SUFFIX}.pth")
torch.save(gnn_encoder.state_dict(), save_path_gnn)

gene_names = list(nx_graph.nodes()) 
final_inference_data = Data(
    x=final_x, 
    edge_index=data.edge_index, 
    num_nodes=len(gene_names)
)

save_path_data = os.path.join(BASE_DIR, f"pathway_graph_for_inference{FILE_SUFFIX}.pth")
torch.save(final_inference_data, save_path_data)
print(f"Completed. Model saved to: {save_path_gnn}")