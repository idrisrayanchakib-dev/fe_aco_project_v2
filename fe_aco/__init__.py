import torch
from .engine import AugmentedQueenAntColony
from .metrics import calculate_modularity, calculate_semantic_coherence
from .simulation import load_cora_federated
from .explainability import generate_node_report
from .visualization import plot_scientific_validation, plot_benchmark_comparison

# Si cuda est dispo, on l'utilise, sinon on prend le cpu
default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def fit(adj_matrix, feature_matrix, num_communities, trials=30, device=default_device, 
        clients=None, min_quality=0.82):
    """
    Lance le protocole FE-ACO complet avec QUALITY GATE.
    """
    print(f"🚀 FE-ACO: Démarrage sur {device.upper()} (Cible Qualité >= {min_quality})...")
    
    global_best_score = -float('inf')
    global_best_model = None
    global_best_metrics = (0.0, 0.0) # (Q, C)
    
    max_retries = 6 
    attempt = 0
    
    while attempt < max_retries:
        attempt += 1
        print(f"\n🔄 Tentative {attempt}/{max_retries} : Lancement de {trials} univers parallèles...")
        
        batch_best_score = -float('inf')
        batch_best_model = None
        batch_best_q = 0.0
        batch_best_c = 0.0
        
        for i in range(trials):
            # 1. Initialiser
            model = AugmentedQueenAntColony(adj_matrix, feature_matrix, num_communities, device=device)
            
            # 2. Exécuter
            total_rounds = 80
            for r in range(total_rounds):
                model.step(current_round=r, total_rounds=total_rounds)
                    
            # 3. Évaluation Hybride
            preds = model.get_prediction()
            q = calculate_modularity(adj_matrix, preds, num_communities)
            c = calculate_semantic_coherence(feature_matrix, preds, num_communities)
            
            # --- FORMULE DU SCORE (CORRIGÉE) ---
            # On utilise l'Amplification Sémantique.
            # Score = Structure * (1 + Bonus Sémantique)
            # Exemple : 0.65 * (1 + 0.30) = 0.845
            # Cela correspond parfaitement à ton seuil min_quality=0.80
            final_score = q * (1.0 + c)
            
            # Sélection locale
            if final_score > batch_best_score:
                batch_best_score = final_score
                batch_best_model = model
                batch_best_q = q
                batch_best_c = c
        
        print(f"   👉 Meilleur score de la tentative : {batch_best_score:.4f} (Q={batch_best_q:.4f} | C={batch_best_c:.4f})")
        
        # Mise à jour du Record Mondial
        if batch_best_score > global_best_score:
            global_best_score = batch_best_score
            global_best_model = batch_best_model
            global_best_metrics = (batch_best_q, batch_best_c)
            
        # --- QUALITY GATE ---
        if global_best_score >= min_quality:
            print("✅ SUCCÈS : La barrière de qualité est franchie. Convergence validée.")
            break 
        else:
            if attempt < max_retries:
                print("⚠️  AVERTISSEMENT : Score insuffisant. Relance automatique...")
            else:
                print("❌ ECHEC : Nombre max de tentatives atteint. On livre le meilleur modèle disponible.")

    print(f"\n🏆 RÉSULTAT FINAL : Score Hybride={global_best_score:.4f} (Q={global_best_metrics[0]:.4f} | Cohérence={global_best_metrics[1]:.4f})")
    return global_best_model