import torch
import numpy as np

def generate_node_report(node_id, model, adj_original, dataset_labels=None):
    """
    Génère un rapport d'explicabilité (Transparence).
    Retourne un dictionnaire.
    """
    # 1. Décision du modèle
    pheromones = model.get_confidence_vector()[node_id].detach().cpu().numpy()
    chosen_comm = np.argmax(pheromones)
    confidence = pheromones[chosen_comm]
    
    # 2. Analyse locale
    if adj_original.is_sparse:
        adj_original = adj_original.to_dense()
        
    neighbors = torch.nonzero(adj_original[node_id]).flatten().cpu().numpy()
    degree = len(neighbors)
    
    # 3. Logique d'Explication (Ton Innovation)
    analysis = "UNKNOWN"
    if degree == 0:
        analysis = "ISOLATED NODE (Saved by Semantic Augmentation)"
    elif confidence > 0.95:
        if degree > 10:
            analysis = "DOMINANT HUB (Core Anchor)"
        elif degree < 3:
            analysis = "LEAF NODE (Follower)"
        else:
            analysis = "CORE MEMBER"
    elif confidence > 0.60:
        analysis = "PERIPHERAL NODE"
    else:
        analysis = "BRIDGE / BOUNDARY (Ambiguous)"
        
    report = {
        "node_id": int(node_id),
        "cluster_id": int(chosen_comm),
        "confidence": float(confidence),
        "degree": int(degree),
        "analysis": analysis
    }
    
    if dataset_labels is not None:
        true_label = dataset_labels[node_id].item()
        report["true_label"] = int(true_label)
        
    return report