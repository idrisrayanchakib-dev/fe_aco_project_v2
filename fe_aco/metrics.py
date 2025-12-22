import torch
import torch.nn.functional as F

def calculate_modularity(adj, pred_labels, num_communities):
    """
    Calcule la Modularité (Q) sur le GPU.
    Utilisé pour la sélection sans vérité terrain.
    """
    m = adj.sum()
    if m == 0: return 0.0
    
    # Gestion Sparse/Dense pour le calcul
    if adj.is_sparse:
        adj = adj.to_dense() # Pour <50k nœuds ça passe en mémoire

    k = adj.sum(dim=1).view(-1, 1)
    clusters = F.one_hot(pred_labels, num_classes=num_communities).float()
    
    expected_adj = torch.mm(k, k.t()) / m
    B = adj - expected_adj
    
    q_matrix = torch.mm(clusters.t(), torch.mm(B, clusters))
    Q = torch.trace(q_matrix) / m
    return Q.item()

def calculate_semantic_coherence(feature_matrix, pred_labels, num_communities):
    """
    Calcule à quel point les clusters sont 'purs' sémantiquement.
    On calcule la similarité moyenne au centre du cluster.
    """
    total_coherence = 0.0
    
    # Pour chaque communauté
    for c in range(num_communities):
        # Trouver les nœuds qui appartiennent à ce cluster
        mask = (pred_labels == c)
        
        # S'il n'y a personne ou un seul nœud, on ignore
        if mask.sum() <= 1: 
            continue
            
        # Récupérer les features de ce cluster
        cluster_feats = feature_matrix[mask]
        
        # Calculer le "Centre" (Centroïde) du cluster
        centroid = cluster_feats.mean(dim=0, keepdim=True)
        centroid = F.normalize(centroid, p=2, dim=1)
        
        # Normaliser les membres
        cluster_feats = F.normalize(cluster_feats, p=2, dim=1)
        
        # Calculer la similarité de chaque membre avec le centre
        sims = torch.mm(cluster_feats, centroid.t())
        
        # La cohérence est la moyenne des similarités
        total_coherence += sims.mean().item()
        
    # On fait la moyenne sur tous les clusters
    return total_coherence / num_communities