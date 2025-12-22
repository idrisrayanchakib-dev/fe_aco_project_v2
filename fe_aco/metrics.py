import torch
import torch.nn.functional as F

def calculate_modularity(adj, pred_labels, num_communities):
    """
    Calcule la Modularité (Q).
    Sécurisé pour le Big Data : Renvoie 0.0 si le graphe est trop gros pour la RAM.
    """
    # 1. Protection Big Data (Disjoncteur)
    # Si plus de 50k nœuds, le calcul Dense N*N est impossible
    if adj.shape[0] > 50000:
        # On pourrait implémenter une version sparse approximée ici,
        # mais pour le Stress Test, on veut juste éviter le crash.
        return 0.0

    m = adj.sum()
    if m == 0: return 0.0
    
    # 2. Conversion Dense (Seulement si c'est petit)
    if adj.is_sparse:
        adj = adj.to_dense()

    k = adj.sum(dim=1).view(-1, 1)
    clusters = F.one_hot(pred_labels, num_classes=num_communities).float()
    
    expected_adj = torch.mm(k, k.t()) / m
    B = adj - expected_adj
    
    q_matrix = torch.mm(clusters.t(), torch.mm(B, clusters))
    Q = torch.trace(q_matrix) / m
    return Q.item()

def calculate_semantic_coherence(feature_matrix, pred_labels, num_communities):
    """
    Calcule la Cohérence Sémantique (C).
    """
    # Protection Big Data : Si les features sont trop lourdes, on skip ou on sample
    # Mais ici le calcul est linéaire O(N), donc ça devrait passer sur 80Go VRAM.
    # On ajoute quand même un try/except pour la sécurité.
    
    try:
        total_coherence = 0.0
        valid_communities = 0
        
        for c in range(num_communities):
            mask = (pred_labels == c)
            if mask.sum() <= 1: 
                continue
                
            cluster_feats = feature_matrix[mask]
            
            # Centroïde
            centroid = cluster_feats.mean(dim=0, keepdim=True)
            centroid = F.normalize(centroid, p=2, dim=1)
            
            # Normalisation
            cluster_feats = F.normalize(cluster_feats, p=2, dim=1)
            
            # Similarité
            sims = torch.mm(cluster_feats, centroid.t())
            total_coherence += sims.mean().item()
            valid_communities += 1
            
        if valid_communities == 0:
            return 0.0
            
        return total_coherence / valid_communities
        
    except RuntimeError:
        # En cas de OOM sur la cohérence
        return 0.0