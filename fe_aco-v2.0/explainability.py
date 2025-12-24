import torch
import numpy as np

def generate_node_report(node_id, model, adj_original, node_names=None, node_types=None):
    """
    Génère un rapport d'explicabilité Clinique & Pharmacologique.
    Optimisé pour les matrices Sparse (Zero-Copy).
    
    Args:
        node_id (int): L'index du nœud.
        model (AugmentedQueenAntColony): Le modèle entraîné.
        adj_original (torch.Tensor): Matrice d'adjacence (Sparse).
        node_names (dict, optional): Map {id: "BRCA1"}.
        node_types (dict, optional): Map {id: "Protein" ou "Drug"}.
    """
    # 1. Décision du modèle (Phéromones)
    pheromones = model.get_confidence_vector()[node_id].detach().cpu().numpy()
    chosen_comm = np.argmax(pheromones)
    confidence = pheromones[chosen_comm]
    
    # 2. Analyse Topologique (SPARSE SAFE)
    # INTERDIT : adj_original.to_dense() sur H100 avec gros graphe
    degree = 0
    if adj_original.is_sparse:
        # On compte les occurrences du node_id dans les indices row
        # C'est beaucoup plus léger que de densifier
        indices = adj_original.indices()
        # Masque booléen sur les indices (rapide sur GPU)
        degree = (indices[0] == node_id).sum().item()
    else:
        degree = torch.count_nonzero(adj_original[node_id]).item()
    
    # 3. Logique d'Explication "SMART ADN" (Drug Repurposing Logic)
    analysis = "UNKNOWN"
    node_type = node_types.get(node_id, "Unknown") if node_types else "Entity"
    
    # Cas A : Le "Coup de Génie" (Zéro lien connu, mais forte confiance IA)
    # C'est le cas du Drug Repurposing pur
    if degree == 0:
        if confidence > 0.85:
            analysis = "NOVEL CANDIDATE (Pure Semantic Discovery)"
        else:
            analysis = "ISOLATED ENTITY (Low Evidence)"
            
    # Cas B : La Cible/Molécule "Star" (Hub)
    elif degree > 20:
        analysis = "WELL-ESTABLISHED (High Connectivity Hub)"
        
    # Cas C : Le Candidat Solide
    elif confidence > 0.90:
        analysis = "STRONG PREDICTION (High Confidence)"
        
    # Cas D : L'Ambiguïté (Poly-pharmacologie potentielle)
    elif confidence < 0.55:
        analysis = "AMBIGUOUS / MULTI-TARGET POTENTIAL"
    else:
        analysis = "PERIPHERAL CONNECTION"
        
    # 4. Construction du Rapport
    name = str(node_id)
    if node_names and node_id in node_names:
        name = node_names[node_id]

    report = {
        "node_id": int(node_id),
        "name": name,
        "type": node_type,
        "cluster_id": int(chosen_comm),
        "ai_confidence_score": float(confidence),
        "known_interactions_count": int(degree),
        "clinical_interpretation": analysis
    }
    
    return report