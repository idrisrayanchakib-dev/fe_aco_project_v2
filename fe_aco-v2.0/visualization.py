import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch

def plot_scientific_validation(results, save_path="validation_metrics.png"):
    """
    Validation Technique : Modularity vs Coherence
    (On a remplacé NMI par Coherence car on n'a pas toujours la vérité terrain)
    """
    modularity_scores = [r[1] for r in results]
    coherence_scores = [r[2] for r in results] # C'était NMI, c'est maintenant Coherence

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
    print(f"📊 Validation Plot saved: {save_path}")

def plot_drug_discovery_cluster(adj, model, target_node_id, node_types, node_names=None, save_path="drug_cluster.png"):
    """
    MONEY SHOT : Visualisation du Cluster Thérapeutique.
    Affiche la protéine cible et les médicaments connectés par l'IA.
    """
    print(f"💊 Génération du graphe pour la cible #{target_node_id}...")
    
    # 1. Identifier le cluster de la cible
    preds = model.get_prediction()
    target_cluster = preds[target_node_id].item()
    
    # 2. Extraire les membres du cluster (Masque)
    # Sur H100, on reste sur CPU pour le dessin
    cluster_mask = (preds == target_cluster).cpu()
    node_indices = torch.nonzero(cluster_mask).flatten()
    
    # Si le cluster est trop gros (>100 nœuds), on ne garde que le top confiance
    if len(node_indices) > 50:
        confidences = model.get_confidence_vector()[node_indices, target_cluster]
        _, top_idx = torch.topk(confidences, k=50)
        node_indices = node_indices[top_idx]
        
    node_indices_list = node_indices.tolist()
    
    # 3. Extraire le Sous-Graphe (Subgraph)
    # On convertit uniquement cette petite partie en NetworkX
    # (Attention : adj doit être sur CPU ici)
    if adj.is_sparse:
        # Extraction manuelle ou via torch_geometric si dispo
        # Ici on fait une version simplifiée : on ne dessine pas toutes les arêtes internes,
        # juste les liens forts ou sémantiques. Pour la beauté, on peut juste dessiner les nœuds.
        G = nx.Graph()
        for idx in node_indices_list:
            G.add_node(idx)
    else:
        # Si dense (petit graphe), facile
        sub_adj = adj[node_indices][:, node_indices].cpu().numpy()
        G = nx.from_numpy_array(sub_adj)
        # Remapping des labels
        mapping = {i: node_indices_list[i] for i in range(len(node_indices_list))}
        G = nx.relabel_nodes(G, mapping)

    # 4. Couleurs et Labels
    colors = []
    labels = {}
    sizes = []
    
    for node in G.nodes():
        n_type = node_types[node].item()
        is_target = (node == target_node_id)
        
        # Labels
        if node_names and node in node_names:
            labels[node] = node_names[node]
        else:
            labels[node] = str(node) if is_target or n_type == 2 else ""

        # Couleurs
        if is_target:
            colors.append('#FF0000') # Rouge = Cible
            sizes.append(3000)
        elif n_type == 2: # Médicament
            colors.append('#00FF00') # Vert = Drogue
            sizes.append(1500)
        else:
            colors.append('#CCCCCC') # Gris = Autre Protéine
            sizes.append(500)
            
    # 5. Dessin
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.3, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, edgecolors='k')
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    plt.title(f"AI-Predicted Therapeutic Cluster for Target #{target_node_id}", fontsize=15)
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💊 Drug Cluster Graph saved: {save_path}")

def plot_discovery_matrix(candidates, save_path="discovery_matrix.png"):
    """
    Visualise les candidats selon Confiance vs Nouveauté.
    Axe X : Degré connu (Known interactions)
    Axe Y : Confiance IA (AI Confidence)
    """
    degrees = [c['known_interactions_count'] for c in candidates]
    confidences = [c['ai_confidence_score'] for c in candidates]
    names = [c['name'] for c in candidates]
    
    plt.figure(figsize=(10, 8))
    
    # Zone "NOVEL HITS" : Degré faible, Confiance haute
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
    print(f"💎 Discovery Matrix saved: {save_path}")