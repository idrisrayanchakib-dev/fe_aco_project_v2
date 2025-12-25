import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch

def plot_scientific_validation(results, save_path="validation_metrics.png"):
    """
    Validation Technique : Modularity vs Coherence
    """
    modularity_scores = [r[1] for r in results]
    coherence_scores = [r[2] for r in results]

    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Scatter plot avec code couleur selon le score final
    scores = [m * (1+c) for m, c in zip(modularity_scores, coherence_scores)]
    plt.scatter(modularity_scores, coherence_scores, c=scores, cmap='viridis', s=100, edgecolors='k')
    plt.colorbar(label='Hybrid Score (Optimization Objective)')
    
    plt.xlabel('Structural Modularity (Q)')
    plt.ylabel('Therapeutic Coherence (C)')
    plt.title('Convergence of Parallel Universes (FE-ACO)')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # Important pour libérer la mémoire
    print(f"📊 Validation Plot saved: {save_path}")

def plot_drug_discovery_cluster(adj, model, target_node_id, node_types, node_names=None, save_path="drug_cluster.png"):
    """
    MONEY SHOT : Visualisation du Cluster Thérapeutique.
    Version CPU-Safe pour éviter les erreurs de device.
    """
    print(f"💊 Génération du graphe pour la cible #{target_node_id}...")
    
    # --- 1. TRANSFERT CPU SÉCURISÉ (Le Correctif) ---
    # On détache tout du GPU pour éviter les conflits avec Matplotlib/NetworkX
    preds = model.get_prediction().detach().cpu()
    conf_matrix = model.get_confidence_vector().detach().cpu()
    
    # Gestion de node_types si c'est un tenseur
    if isinstance(node_types, torch.Tensor):
        node_types = node_types.cpu()

    # 2. Identifier le cluster de la cible
    target_cluster = preds[target_node_id].item()
    
    # 3. Extraire les membres du cluster
    cluster_mask = (preds == target_cluster)
    node_indices = torch.nonzero(cluster_mask).flatten()
    
    # Si le cluster est trop gros (>50 nœuds), on ne garde que le top confiance
    # pour que le dessin reste lisible
    if len(node_indices) > 50:
        # Tout est sur CPU ici, donc pas d'erreur
        confidences = conf_matrix[node_indices, target_cluster]
        _, top_idx = torch.topk(confidences, k=50)
        node_indices = node_indices[top_idx]
        
    # On s'assure que la cible est bien dans la liste
    if target_node_id not in node_indices:
        node_indices = torch.cat([torch.tensor([target_node_id]), node_indices])
        
    node_indices_list = node_indices.tolist()
    
    # 4. Construction du Graphe NetworkX
    G = nx.Graph()
    for idx in node_indices_list:
        G.add_node(idx)
        
    # Ajout des arêtes sémantiques (Visualisation en étoile)
    # On connecte la cible aux médicaments si la confiance est forte
    for idx in node_indices_list:
        if idx != target_node_id:
            score = conf_matrix[idx, target_cluster].item()
            if score > 0.8: # Seuil visuel
                G.add_edge(target_node_id, idx)

    # 5. Couleurs et Labels
    colors = []
    labels = {}
    sizes = []
    
    for node in G.nodes():
        # Lecture safe du type
        if isinstance(node_types, torch.Tensor):
            n_type = node_types[node].item()
        else:
            n_type = node_types[node] # Si c'est un dict ou une liste
            
        is_target = (node == target_node_id)
        
        # Labels
        if node_names and node in node_names:
            clean_name = str(node_names[node])
            # Troncation si trop long
            labels[node] = clean_name if len(clean_name) < 12 else clean_name[:9]+"..."
        else:
            labels[node] = str(node) if is_target or n_type == 2 else ""

        # Couleurs
        if is_target:
            colors.append('#FF4444') # Rouge = Cible
            sizes.append(2500)
        elif n_type == 2: # Médicament
            colors.append('#00CC00') # Vert = Drogue
            sizes.append(1000)
        else:
            colors.append('#CCCCCC') # Gris = Autre
            sizes.append(300)
            
    # 6. Dessin
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.25, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, edgecolors='white')
    nx.draw_networkx_edges(G, pos, edge_color='#E0E0E0', alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
    
    plt.title(f"AI-Predicted Cluster for Target #{target_node_id}", fontsize=15)
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💊 Drug Cluster Graph saved: {save_path}")

def plot_discovery_matrix(candidates, save_path="discovery_matrix.png"):
    """
    Visualise les candidats selon Confiance vs Nouveauté.
    """
    degrees = [c['known_interactions_count'] for c in candidates]
    confidences = [c['ai_confidence_score'] for c in candidates]
    names = [c['name'] for c in candidates]
    
    plt.figure(figsize=(10, 8))
    
    # Zone "NOVEL HITS"
    plt.axvspan(-0.5, 2.5, ymin=0.8, ymax=1.0, color='green', alpha=0.1, label='Novel Hits Zone')
    
    plt.scatter(degrees, confidences, c='blue', s=100, alpha=0.7)
    
    for i, txt in enumerate(names):
        plt.annotate(txt, (degrees[i], confidences[i]), xytext=(5, 5), textcoords='offset points')
        
    plt.xlabel('Known Interactions (Degree)')
    plt.ylabel('AI Confidence Score')
    plt.title('Drug Repurposing Discovery Matrix')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💎 Discovery Matrix saved: {save_path}")