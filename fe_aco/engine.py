import torch
import torch.nn.functional as F

class AugmentedQueenAntColony:
    def __init__(self, adj_matrix, feature_matrix, num_communities, device='cuda', 
                 aug_weight=0.5, top_k=50): 
        """
        FE-ACO ENGINE V2.0 (H100/Biomedical Edition)
        Optimisé pour les graphes massifs et l'intégration GNN.
        Gestion mémoire par 'Chunking' et opérations Sparse intégrales.
        """
        self.device = device
        self.k = num_communities
        self.num_nodes = adj_matrix.shape[0]
        
        # S'assurer que la matrice d'adjacence est Sparse dès le début
        if not adj_matrix.is_sparse:
            adj_matrix = adj_matrix.to_sparse()
        
        # --- FIX STABILITÉ : Coalesce obligatoire sur GPU ---
        adj_matrix = adj_matrix.coalesce()
        
        # --- 1. GRAPH AUGMENTATION (MEMORY EFFICIENT) ---
        # Normalisation des features (Embeddings GNN)
        feat_norm = F.normalize(feature_matrix, p=2, dim=1)
        
        # Calcul de similarité par blocs (évite l'OOM sur H100)
        # On récupère directement une matrice sparse coalesced
        self.sparse_sim = self._compute_sparse_similarity(feat_norm, top_k=top_k)
        
        # --- 2. ATTENTION DYNAMIQUE (Sparse Friendly) ---
        # Calcul du degré sur la matrice sparse
        degrees = torch.sparse.sum(adj_matrix, dim=1).to_dense().view(-1, 1)
        
        # Formule d'attention : Boost inverse au degré
        dynamic_weight = aug_weight + (2.0 / (degrees + 1.0))
        
        # Application du poids dynamique sur la matrice sémantique sparse
        # On multiplie les valeurs de la sparse matrix par le poids du nœud source (row)
        rows = self.sparse_sim.indices()[0]
        weights_per_edge = dynamic_weight[rows].flatten()
        
        # Création de la matrice sémantique pondérée
        new_values = self.sparse_sim.values() * weights_per_edge
        weighted_semantics = torch.sparse_coo_tensor(
            self.sparse_sim.indices(), 
            new_values, 
            self.sparse_sim.size()
        ).to(device)
        
        # --- 3. FUSION ET NORMALISATION ---
        # Addition de deux matrices sparse (Topologie + Sémantique)
        self.fused_adj = adj_matrix + weighted_semantics
        
        # Ajout des self-loops (Identité Sparse)
        indices_eye = torch.arange(self.num_nodes, device=device).repeat(2, 1)
        values_eye = torch.ones(self.num_nodes, device=device)
        eye_sparse = torch.sparse_coo_tensor(indices_eye, values_eye, (self.num_nodes, self.num_nodes)).to(device)
        
        self.fused_adj = self.fused_adj + eye_sparse
        self.fused_adj = self.fused_adj.coalesce() # Optimisation mémoire
        
        # Normalisation de la matrice fusionnée (Degree Normalization)
        deg = torch.sparse.sum(self.fused_adj, dim=1).to_dense().clamp(min=1.0)
        inv_deg = deg.pow(-1)
        
        # Normalisation Row-wise sur structure Sparse
        # V_new = V_old * (1 / Degree_row)
        rows_fused = self.fused_adj.indices()[0]
        norm_values = self.fused_adj.values() * inv_deg[rows_fused]
        
        # --- FIX CRITIQUE : .coalesce() final pour éviter le crash en step() ---
        self.norm_fused_adj = torch.sparse_coo_tensor(
            self.fused_adj.indices(), 
            norm_values, 
            self.fused_adj.size()
        ).to(device).coalesce()
            
        self._reset_queens()

    def _compute_sparse_similarity(self, features, top_k=50, chunk_size=5000):
        """
        Calcule la similarité Cosinus par blocs pour économiser la VRAM.
        Retourne un tenseur Sparse.
        """
        num_nodes = features.shape[0]
        indices_list = []
        values_list = []
        
        with torch.no_grad(): # Gain mémoire
            for i in range(0, num_nodes, chunk_size):
                end = min(i + chunk_size, num_nodes)
                # Bloc courant (N_chunk x Dim)
                block = features[i:end]
                
                # Similarité : Bloc vs Tout le Monde (N_chunk x N)
                # Sur H100, ce calcul temporaire passe large
                sim_block = torch.mm(block, features.t())
                
                # On ne garde que le Top-K tout de suite
                vals, inds = torch.topk(sim_block, k=top_k, dim=1)
                
                # Ajustement des indices de lignes pour correspondre au graphe global
                row_indices = torch.arange(i, end, device=self.device).view(-1, 1).expand_as(inds)
                
                indices_list.append(torch.stack([row_indices.flatten(), inds.flatten()]))
                values_list.append(vals.flatten())
                
                # Nettoyage explicite (optionnel mais prudent)
                del sim_block, vals, inds, row_indices, block
                
        # Concaténation finale
        all_indices = torch.cat(indices_list, dim=1)
        all_values = torch.cat(values_list)
        
        # --- FIX CRITIQUE : .coalesce() ici aussi ---
        return torch.sparse_coo_tensor(
            all_indices, all_values, (num_nodes, num_nodes)
        ).to(self.device).coalesce()

    def _reset_queens(self):
        # Initialisation "One-Hot-Like" des Reines
        self.global_pheromone = torch.ones(self.num_nodes, self.k, device=self.device) / self.k
        perm = torch.randperm(self.num_nodes, device=self.device) # Optim GPU
        queens = perm[:self.k]
        
        # Reset précis
        self.global_pheromone[queens, :] = 0.0
        self.global_pheromone[queens, torch.arange(self.k, device=self.device)] = 1.0

    def step(self, current_round=0, total_rounds=80):
        """
        Propagation Vectorisée (Version Sparse-Only)
        """
        progress = current_round / total_rounds
        inflation = 1.2 + (pow(progress, 2) * 2.5)
        evaporation = 0.1 + (progress * 0.8) 
        
        # Propagation Sparse-Dense Multiplication (Optimisé CUDA)
        neighbor_influence = torch.sparse.mm(self.norm_fused_adj, self.global_pheromone)
            
        probabilities = torch.pow(neighbor_influence, inflation)
        
        # Pruning
        prune_thresh = 0.01 + (progress * 0.05)
        probabilities[probabilities < prune_thresh] = 0.0
        
        row_sums = probabilities.sum(dim=1, keepdim=True)
        # Sécurité numérique
        probabilities = torch.where(row_sums > 1e-9, probabilities / row_sums, 
                                    torch.ones_like(probabilities) / self.k)
        
        self.global_pheromone = (1 - evaporation) * self.global_pheromone + evaporation * probabilities
        
    def get_prediction(self):
        return torch.argmax(self.global_pheromone, dim=1)
    
    def get_confidence_vector(self):
        return self.global_pheromone