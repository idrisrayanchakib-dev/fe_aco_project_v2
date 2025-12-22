import torch
import torch.nn.functional as F

class AugmentedQueenAntColony:
    def __init__(self, adj_matrix, feature_matrix, num_communities, device='cuda', 
                 aug_weight=0.5, top_k=10):  # <--- REGLAGE EQULIBRÉ (Gold Standard)
        """
        FE-ACO ULTIMATE ENGINE (Big Data Ready)
        Intègre : Safety Bypass pour 100M nodes + Recuit Quadratique.
        """
        self.device = device
        self.k = num_communities
        self.num_nodes = adj_matrix.shape[0]
        
        # --- SÉCURITÉ BIG DATA (Le Fix Anti-Crash) ---
        # Si > 50k nœuds, le calcul N*N de similarité est impossible en RAM.
        # On passe en mode "Graphe Brut" optimisé.
        
        if self.num_nodes > 50000:
            print(f"⚠️ Mode Big Data ({self.num_nodes:,} nœuds) : Augmentation Dense désactivée.")
            
            # On utilise le graphe d'entrée tel quel (supposé déjà optimisé/sparse)
            self.norm_fused_adj = adj_matrix
            
            # On s'assure qu'il est bien Sparse
            if not self.norm_fused_adj.is_sparse:
                self.norm_fused_adj = self.norm_fused_adj.to_sparse()
                
            self.is_sparse = True
            
        else:
            # --- MODE STANDARD (Cora / Recherche) ---
            # On fait l'augmentation sémantique complète (Ta recette magique)
            
            # 1. Normalisation & Similarité
            feat_norm = F.normalize(feature_matrix, p=2, dim=1)
            sim_matrix = torch.mm(feat_norm, feat_norm.t())
            
            # 2. Sparsification
            values, indices = torch.topk(sim_matrix, k=top_k, dim=1)
            sparse_sim = torch.zeros_like(sim_matrix)
            sparse_sim.scatter_(1, indices, values)
            
            # 3. Attention Dynamique
            degrees = adj_matrix.sum(dim=1).view(-1, 1)
            dynamic_weight = aug_weight + (2.0 / (degrees + 1.0))
            
            weighted_semantics = dynamic_weight * sparse_sim
            
            # 4. Fusion
            self.fused_adj = adj_matrix + weighted_semantics
            self.fused_adj = self.fused_adj + torch.eye(self.num_nodes, device=device)
            
            # 5. Normalisation
            deg = self.fused_adj.sum(dim=1).clamp(min=1.0)
            norm_fused_adj = deg.pow(-1).view(-1, 1) * self.fused_adj
            
            self.norm_fused_adj = norm_fused_adj
            self.is_sparse = False

        # Initialisation des Reines
        self._reset_queens()
            
    def _reset_queens(self):
        """Plante des reines aléatoires (Optimisé mémoire)."""
        # Utilise le type de données correct (float32 ou bfloat16 selon le graphe)
        dtype = self.norm_fused_adj.dtype if self.norm_fused_adj.is_sparse else torch.float32
        
        self.global_pheromone = torch.ones(self.num_nodes, self.k, device=self.device, dtype=dtype) / self.k
        
        # Génération indices sur CPU pour éviter OOM sur les très gros tenseurs d'indices
        perm = torch.randperm(self.num_nodes, device='cpu')[:self.k].to(self.device)
        
        for i, node_idx in enumerate(perm):
            self.global_pheromone[node_idx, :] = 0.0
            self.global_pheromone[node_idx, i] = 1.0

    def step(self, current_round=0, total_rounds=80):
        """
        Recuit Quadratique.
        """
        progress = current_round / total_rounds
        
        # Inflation : 1.2 -> 3.7
        inflation = 1.2 + (pow(progress, 2) * 2.5)
        
        # Évaporation : 0.1 -> 0.9
        evaporation = 0.1 + (progress * 0.8) 
        
        # Propagation (Compatible Sparse & Dense)
        if self.is_sparse:
            neighbor_influence = torch.sparse.mm(self.norm_fused_adj, self.global_pheromone)
        else:
            neighbor_influence = torch.mm(self.norm_fused_adj, self.global_pheromone)
            
        probabilities = torch.pow(neighbor_influence, inflation)
        
        # Pruning
        prune_thresh = 0.01 + (progress * 0.05)
        probabilities[probabilities < prune_thresh] = 0.0
        
        # Normalisation Safe
        row_sums = probabilities.sum(dim=1, keepdim=True)
        # Masque pour éviter division par zéro sans créer de NaN
        mask = row_sums > 1e-9
        # On ne met à jour que là où il y a de la somme, le reste reste à 0
        probabilities[mask] = probabilities[mask] / row_sums[mask]
        
        self.global_pheromone = (1 - evaporation) * self.global_pheromone + evaporation * probabilities
        
    def get_prediction(self):
        return torch.argmax(self.global_pheromone, dim=1)
    
    def get_confidence_vector(self):
        return self.global_pheromone