import torch
import torch.nn.functional as F

def calculate_modularity(adj, pred_labels, num_communities):
    """
    Calcule la Modularité (Q) de manière 100% SPARSE.
    Optimisé pour H100 : Jamais de conversion dense.
    Complexité : O(E) au lieu de O(N^2).
    """
    if not adj.is_sparse:
        adj = adj.to_sparse()
        
    m = adj.values().sum() # Poids total (ou 2*E si non pondéré)
    if m == 0: return 0.0
    
    # --- TERME 1 : Fraction des liens INTRA-communautaires ---
    # On travaille uniquement sur la liste des arêtes existantes (indices)
    indices = adj.indices() # [2, E]
    values = adj.values()   # [E]
    
    # On récupère le label de la source et de la destination pour chaque arête
    # pred_labels doit être sur le même device
    labels_src = pred_labels[indices[0]]
    labels_dst = pred_labels[indices[1]]
    
    # Masque : on garde l'arête si source et dest sont dans le même cluster
    mask_intra = (labels_src == labels_dst)
    
    # Somme des poids des arêtes internes
    sum_intra_edges = values[mask_intra].sum()
    term1 = sum_intra_edges / m
    
    # --- TERME 2 : Degrés attendus (Modèle nul) ---
    # Calcul des degrés (Sparse sum)
    # degrees[i] = somme des poids incidents à i
    degrees = torch.sparse.sum(adj, dim=1).to_dense() # [N]
    
    # On somme les degrés par communauté
    # Astuce : Scatter add est ultra rapide sur GPU
    cluster_degrees = torch.zeros(num_communities, device=adj.device)
    cluster_degrees.scatter_add_(0, pred_labels, degrees)
    
    # Formule : Somme_c ( (Deg_c / 2m)^2 )
    # Note : m ici est la somme des poids (équivalent à 2*E pour graphe non dirigé standard)
    term2 = (cluster_degrees / m).pow(2).sum()
    
    Q = term1 - term2
    return Q.item()

def calculate_semantic_coherence(feature_matrix, pred_labels, num_communities):
    """
    Calcule la cohérence des embeddings (GNN features).
    Optimisé pour H100 : Batch processing implicite.
    """
    total_coherence = 0.0
    valid_clusters = 0
    
    for c in range(num_communities):
        mask = (pred_labels == c)
        count = mask.sum().item()
        
        if count <= 1: 
            continue
            
        # Extraction des features du cluster
        # Sur H100, feature_matrix peut être énorme, c'est ok de slicer
        cluster_feats = feature_matrix[mask]
        
        # Centroïde
        centroid = cluster_feats.mean(dim=0, keepdim=True)
        centroid = F.normalize(centroid, p=2, dim=1)
        
        # Normalisation des membres
        cluster_feats = F.normalize(cluster_feats, p=2, dim=1)
        
        # Similarité moyenne
        sims = torch.mm(cluster_feats, centroid.t())
        
        total_coherence += sims.mean().item()
        valid_clusters += 1
        
    if valid_clusters == 0:
        return 0.0
        
    return total_coherence / valid_clusters

def calculate_biological_richness(pred_labels, node_types, num_communities):
    """
    METRIQUE SMART ADN : 
    Vérifie si les clusters contiennent un mix (Protéine + Drogue).
    Si un cluster est 100% drogues ou 100% protéines, il est inutile.
    
    Args:
        node_types (Tensor): [N], 0=Autre, 1=Protéine, 2=Médicament
    Returns:
        float: Score moyen d'entropie biologique (0 à 1)
    """
    score = 0.0
    valid = 0
    
    for c in range(num_communities):
        mask = (pred_labels == c)
        if mask.sum() == 0: continue
            
        # Types présents dans ce cluster
        types_in_cluster = node_types[mask].float()
        
        # On veut idéalement un équilibre. 
        # On peut utiliser la variance ou simplement la présence des deux.
        has_prot = (types_in_cluster == 1).any()
        has_drug = (types_in_cluster == 2).any()
        
        if has_prot and has_drug:
            score += 1.0 # Ce cluster est un candidat valide au criblage
            
        valid += 1
        
    return score / max(valid, 1)