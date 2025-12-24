import torch
import numpy as np

# Imports des modules v2.0
from .engine import AugmentedQueenAntColony
from .metrics import calculate_modularity, calculate_semantic_coherence, calculate_biological_richness
from .simulation import load_biomedical_simulation
from .explainability import generate_node_report
from .visualization import plot_scientific_validation, plot_drug_discovery_cluster, plot_discovery_matrix

# Détection H100
default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_smart_adn_pipeline(num_nodes=50000, num_drugs=5000, num_communities=50, 
                           trials=20, device=default_device):
    """
    PIPELINE PRINCIPAL SMART ADN (H100 EDITION).
    
    Orchestre le flux complet :
    1. Génération/Chargement du Knowledge Graph (Protéines + Médicaments).
    2. Criblage Virtuel via FE-ACO (Optimisation Évolutionnaire).
    3. Identification des Candidats Médicaments.
    4. Génération des Preuves Visuelles (Investisseurs).
    """
    print(f"🧬 SMART ADN: Initialisation du protocole sur {device.upper()}...")
    print(f"   - Target Graph Size: {num_nodes} nodes (Heterogeneous)")
    print(f"   - Drug Library Size: {num_drugs} molecules")
    
    # 1. Chargement / Simulation des Données (H100 Scale)
    # Dans un cas réel, tu remplacerais ça par le chargement de tes embeddings GNN
    adj, features, clients, _, node_types = load_biomedical_simulation(device=device)
    
    # 2. Boucle d'Optimisation "Quality Gate"
    print("\n🚀 Lancement du Criblage Virtuel FE-ACO...")
    
    best_model = None
    best_score = -float('inf')
    history = []
    
    # On lance plusieurs univers parallèles
    for i in range(trials):
        # Initialisation du moteur (Sparse & Chunked interne)
        model = AugmentedQueenAntColony(adj, features, num_communities, device=device, top_k=50)
        
        # Propagation (80 rounds suffisent généralement sur H100 grâce à l'inflation quadratique)
        for r in range(80):
            model.step(current_round=r, total_rounds=80)
            
        # Évaluation Multi-Critères
        preds = model.get_prediction()
        
        # A. Modularité Structurelle (Sparse)
        q = calculate_modularity(adj, preds, num_communities)
        
        # B. Cohérence Pharmacologique (Embeddings GNN)
        c = calculate_semantic_coherence(features, preds, num_communities)
        
        # C. Richesse Biologique (Est-ce qu'on mélange Protéines et Drogues ?)
        b = calculate_biological_richness(preds, node_types, num_communities)
        
        # --- SCORE SMART ADN ---
        # On favorise massivement les clusters qui ont du sens biologique (b) et chimique (c)
        # Si b est faible (clusters purs protéines), le score s'effondre.
        final_score = q * (1.0 + c) * (0.8 + 0.4 * b)
        
        history.append((final_score, q, c, b))
        
        if i % 5 == 0:
            print(f"   [Univers {i+1}/{trials}] Score: {final_score:.4f} (Q={q:.2f} | Coh={c:.2f} | Rich={b:.2f})")
            
        if final_score > best_score:
            best_score = final_score
            best_model = model
            
    print(f"\n🏆 MEILLEUR MODÈLE SÉLECTIONNÉ (Score={best_score:.4f})")
    
    # 3. Phase de Découverte (Drug Repurposing)
    print("\n💊 Analyse des Résultats & Extraction des Candidats...")
    
    # On choisit une cible protéine au hasard pour la démo (ex: un Hub protéique)
    # (Dans ton app, ce serait choisi par le médecin)
    degrees = torch.sparse.sum(adj, dim=1).to_dense()
    # On prend une protéine (type 1) avec un fort degré
    is_prot = (node_types == 1)
    target_candidates = torch.nonzero(is_prot & (degrees > 10)).flatten()
    if len(target_candidates) > 0:
        target_id = target_candidates[0].item() # On prend la première
    else:
        target_id = 0 # Fallback
        
    # Génération du rapport pour cette cible
    # Note : node_names serait remplacé par ton dictionnaire {id: "BRCA1", ...}
    report = generate_node_report(target_id, best_model, adj, 
                                  node_types={i: ("Drug" if t==2 else "Protein") for i, t in enumerate(node_types)})
    
    print(f"   - Cible Analysée: Node #{target_id} ({report['clinical_interpretation']})")
    print(f"   - Cluster Assigné: #{report['cluster_id']}")
    
    # 4. Génération des Visuels (Sortie Fichier)
    print("\n🎨 Génération des preuves visuelles...")
    
    # Plot 1: Validation Technique
    plot_scientific_validation(history, save_path="smart_adn_validation.png")
    
    # Plot 2: Le Cluster Thérapeutique (Money Shot)
    plot_drug_discovery_cluster(adj, best_model, target_id, node_types, save_path="smart_adn_target_cluster.png")
    
    # Plot 3: Matrice de Découverte (Pour tout le cluster)
    # On récupère tous les membres du cluster de la cible
    preds = best_model.get_prediction()
    cluster_mask = (preds == report['cluster_id'])
    cluster_members = torch.nonzero(cluster_mask).flatten().tolist()
    
    # On génère les rapports pour les médicaments de ce cluster
    drug_candidates = []
    for member_id in cluster_members:
        if node_types[member_id] == 2: # Si c'est un médicament
            cand_report = generate_node_report(member_id, best_model, adj, 
                                               node_names={member_id: f"Drug_{member_id}"},
                                               node_types={i: ("Drug" if t==2 else "Protein") for i, t in enumerate(node_types)})
            drug_candidates.append(cand_report)
            
    if drug_candidates:
        plot_discovery_matrix(drug_candidates, save_path="smart_adn_discovery_matrix.png")
        print(f"   - {len(drug_candidates)} médicaments candidats identifiés dans ce cluster.")
    
    print("\n✅ PIPELINE TERMINÉ. Prêt pour l'analyse clinique.")
    return best_model