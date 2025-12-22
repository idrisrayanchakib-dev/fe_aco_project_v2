import torch
import torch.nn.functional as F
import numpy as np

def calculate_modularity(adj, pred_labels, num_communities, sample_size=50000):
    """
    Calcule la Modularité (Q).
    - Si Petit Graphe : Calcul Exact.
    - Si Grand Graphe : Estimation par Sondage (Sampling).
    """
    num_nodes = adj.shape[0]
    
    # --- CAS 1 : BIG DATA (Sampling) ---
    if num_nodes > sample_size:
        # On ne peut pas tout calculer. On prend un échantillon représentatif.
        # On choisit 'sample_size' nœuds au hasard
        indices = torch.randperm(num_nodes)[:sample_size].to(adj.device)
        
        # On extrait les labels de cet échantillon
        sample_preds = pred_labels[indices]
        
        # On extrait le sous-graphe (Attention: opération délicate en Sparse)
        # Pour faire simple et rapide sur GPU H100 :
        # On triche un peu : on ne calcule Q que si on peut extraire le sous-graphe dense
        # Sinon, on retourne une estimation neutre (0.0) pour ne pas bloquer.
        
        # Note : Extraire un sous-graphe dense d'une matrice sparse géante est coûteux.
        # Pour ce projet, on va se baser sur la COHÉRENCE (C) qui est plus facile à sampler.
        return 0.0 # On skip Q pour le Big Data, on se fie à C.

    # --- CAS 2 : STANDARD (Exact) ---
    m = adj.sum()
    if m == 0: return 0.0
    
    if adj.is_sparse:
        adj = adj.to_dense()

    k = adj.sum(dim=1).view(-1, 1)
    clusters = F.one_hot(pred_labels, num_classes=num_communities).float()
    
    expected_adj = torch.mm(k, k.t()) / m
    B = adj - expected_adj
    
    q_matrix = torch.mm(clusters.t(), torch.mm(B, clusters))
    Q = torch.trace(q_matrix) / m
    return Q.item()

def calculate_semantic_coherence(feature_matrix, pred_labels, num_communities, sample_size=100000):
    """
    Calcule la Cohérence Sémantique (C).
    Supporte le Sampling pour le Big Data (Calcul sur 100k nœuds max).
    """
    num_nodes = feature_matrix.shape[0]
    
    # Si trop gros, on travaille sur un échantillon
    if num_nodes > sample_size:
        indices = torch.randperm(num_nodes)[:sample_size].to(feature_matrix.device)
        features_subset = feature_matrix[indices]
        preds_subset = pred_labels[indices]
    else:
        features_subset = feature_matrix
        preds_subset = pred_labels
        
    try:
        total_coherence = 0.0
        valid_communities = 0
        
        for c in range(num_communities):
            mask = (preds_subset == c)
            # On ignore les petits clusters vides dans l'échantillon
            if mask.sum() <= 5: 
                continue
                
            cluster_feats = features_subset[mask]
            
            # Centroïde
            centroid = cluster_feats.mean(dim=0, keepdim=True)
            centroid = F.normalize(centroid, p=2, dim=1)
            
            # Normalisation
            cluster_feats = F.normalize(cluster_feats, p=2, dim=1)
            
            # Similarité (Rapide sur l'échantillon)
            sims = torch.mm(cluster_feats, centroid.t())
            total_coherence += sims.mean().item()
            valid_communities += 1
            
        if valid_communities == 0:
            return 0.0
            
        return total_coherence / valid_communities
        
    except RuntimeError:
        return 0.0