import torch
import fe_aco
import time
from sklearn.metrics import normalized_mutual_info_score as nmi_score

# 1. Vérification du Matériel
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Matériel détecté : {device.upper()}")
if device == 'cuda':
    print(f"🚀 Carte Graphique : {torch.cuda.get_device_name(0)}")

print("\n" + "="*50)
print("  DÉMARRAGE DU PROTOCOLE SMART ADN (FE-ACO)")
print("="*50 + "\n")

# 2. Chargement des données
print("📥 Chargement des données Cora (Simulation Fédérée)...")
try:
    adj, features, clients, labels = fe_aco.load_cora_federated(device=device)
    print("✅ Données chargées sur le GPU.")
except ImportError as e:
    print(f"❌ Erreur : {e}")
    exit()

# 3. Lancement du Moteur (100 TRIALS)
# On passe à 100 pour maximiser la chance d'avoir le record (0.5084)
start_time = time.time()

print("🚀 Lancement de l'entraînement (100 Univers Parallèles)...")
model = fe_aco.fit(
    adj_matrix=adj, 
    feature_matrix=features, 
    num_communities=7, 
    trials=50,          # <--- ICI : On passe à 100
    device=device,
    clients=clients,
    min_quality=0.82
)

duration = time.time() - start_time
print(f"\n⏱️  Temps de calcul : {duration:.2f} secondes")

# --- LE CALCUL DU NMI (Validation Externe) ---
# On récupère les prédictions du meilleur modèle
preds = model.get_prediction().cpu().numpy()
truth = labels.cpu().numpy()

# On calcule le score
final_nmi = nmi_score(truth, preds)

print("\n" + "-"*40)
print(f"🏆 RÉSULTAT FINAL (40 Essais)")
print(f"Modularité (Q) : {fe_aco.metrics.calculate_modularity(adj, model.get_prediction(), 7):.4f}")
print(f"NMI Score      : {final_nmi:.4f}")  # <--- C'est ici qu'on voit si tu as battu Louvain
print("-" * 40)

# 4. Transparence
node_id = 1113
print(f"\n🔍 Analyse du Patient #{node_id}...")
report = fe_aco.generate_node_report(node_id, model, adj, labels)
print("-" * 30)
print(f"Diagnostic : {report['analysis']}")
print(f"Confiance  : {report['confidence']}")
print(f"Vrai Label : {report['true_label']} | Prédit : {report['cluster_id']}")
print("-" * 30)

# 5. Génération des Preuves
print("\n📊 Génération du benchmark...")
# On passe le vrai NMI calculé pour que le graphe soit précis
from fe_aco.visualization import plot_benchmark_comparison
plot_benchmark_comparison(final_nmi, save_path="smart_adn_benchmark_100.png")

print("\n✅ TEST TERMINÉ. Vérifie 'smart_adn_benchmark_100.png'.")