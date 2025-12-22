import torch

def load_cora_federated(device='cuda'):
    """
    Charge Cora et simule la séparation fédérée (Privacy Masking).
    Nécessite torch_geometric.
    """
    try:
        from torch_geometric.datasets import Planetoid
        from torch_geometric.utils import to_dense_adj
    except ImportError:
        raise ImportError("Installez torch_geometric pour utiliser cette fonction.")

    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0].to(device)
    
    adj = to_dense_adj(data.edge_index)[0]
    adj = torch.max(adj, adj.t()) # Symmetrization
    
    num_nodes = data.num_nodes
    split_size = num_nodes // 3
    clients = []
    
    for i in range(3):
        start = i * split_size
        end = (i + 1) * split_size if i < 2 else num_nodes
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        mask[start:end] = True
        
        # PRIVACY MASKING (La preuve !)
        client_adj = adj.clone()
        client_adj[~mask, :] = 0
        
        clients.append({'id': i, 'mask': mask, 'adj': client_adj})
        
    return adj, data.x, clients, data.y