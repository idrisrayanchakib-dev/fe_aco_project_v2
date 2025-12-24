import torch

def generate_biomedical_graph(num_nodes=50000, num_drugs=5000, device='cuda'):
    """
    Génère un Graphe Biomédical Synthétique (Protéines + Médicaments).
    Structure réaliste : PPI (Protein-Protein Interaction) + DTI (Drug-Target Interaction).
    """
    print(f"🧪 Génération d'un graphe biomédical synthétique ({num_nodes} nœuds)...")
    
    num_proteins = num_nodes - num_drugs
    
    # 1. Node Types (1 = Protéine, 2 = Médicament)
    node_types = torch.ones(num_nodes, device=device)
    # Les derniers indices sont les médicaments
    drug_start_idx = num_proteins
    node_types[drug_start_idx:] = 2 
    
    # 2. Génération des Arêtes (Approche Sparse aléatoire optimisée)
    # A. Protein-Protein Interactions (PPI) - Dense
    # On génère des liens aléatoires entre protéines
    num_ppi = num_proteins * 10 # Moyenne de 10 voisins
    src_p = torch.randint(0, num_proteins, (num_ppi,), device=device)
    dst_p = torch.randint(0, num_proteins, (num_ppi,), device=device)
    
    # B. Drug-Target Interactions (DTI) - Sparse mais critique
    # Chaque médicament cible quelques protéines
    num_dti = num_drugs * 3 # Moyenne de 3 cibles par drogue
    src_d = torch.randint(drug_start_idx, num_nodes, (num_dti,), device=device)
    dst_d = torch.randint(0, num_proteins, (num_dti,), device=device) # Cible une protéine
    
    # Fusion des arêtes
    # On rend le graphe non-dirigé : (src->dst) + (dst->src)
    src = torch.cat([src_p, src_d, dst_p, dst_d])
    dst = torch.cat([dst_p, dst_d, src_p, src_d])
    
    indices = torch.stack([src, dst])
    values = torch.ones(indices.shape[1], device=device)
    
    # Création Matrice Sparse
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()
    
    # 3. Features Synthétiques (Simulation Embeddings GNN)
    # Dim 128 (Classique pour GNN)
    features = torch.randn(num_nodes, 128, device=device)
    
    return adj, features, node_types

def simulate_federated_split(adj, num_clients=3, device='cuda'):
    """
    Simule une séparation fédérée (Hôpitaux) en mode 100% SPARSE.
    Zéro conversion dense = Scalabilité infinie sur H100.
    """
    print(f"🏥 Simulation de la fédération en {num_clients} silos isolés...")
    
    num_nodes = adj.shape[0]
    split_size = num_nodes // num_clients
    clients = []
    
    # Extraction des indices globaux une seule fois
    indices = adj.indices()
    values = adj.values()
    row_indices = indices[0]
    
    for i in range(num_clients):
        start = i * split_size
        end = (i + 1) * split_size if i < num_clients - 1 else num_nodes
        
        # PRIVACY MASKING (Version Sparse)
        # Un client ne voit que les liens qui partent de ses nœuds (Lignes)
        # Il ne voit pas les liens internes aux autres hôpitaux.
        
        # Masque booléen sur les indices (Ultra rapide)
        # On garde l'arête SI source est dans [start, end]
        mask = (row_indices >= start) & (row_indices < end)
        
        client_indices = indices[:, mask]
        client_values = values[mask]
        
        # On recrée un tenseur sparse propre pour ce client
        # Note : La taille globale reste (N, N) pour garder la cohérence des ID
        client_adj = torch.sparse_coo_tensor(
            client_indices, 
            client_values, 
            (num_nodes, num_nodes)
        ).coalesce()
        
        clients.append({
            'id': i,
            'range': (start, end),
            'adj': client_adj
        })
        
    return clients

def load_biomedical_simulation(device='cuda'):
    """
    Fonction principale à appeler dans __init__.py
    """
    # Génération
    adj, features, node_types = generate_biomedical_graph(device=device)
    
    # Fédération
    clients = simulate_federated_split(adj, device=device)
    
    # On simule une "Ground Truth" (Communautés) pour l'évaluation technique
    # Dans la vraie vie, on ne l'a pas. Ici on génère des labels aléatoires pour tester le pipeline.
    # Pour le Drug Repurposing, on s'en fiche un peu des labels, on veut les connexions.
    num_nodes = adj.shape[0]
    fake_labels = torch.randint(0, 50, (num_nodes,), device=device)
    
    return adj, features, clients, fake_labels, node_types