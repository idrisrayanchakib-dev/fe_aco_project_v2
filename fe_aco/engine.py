import torch
import torch.nn.functional as F

class AugmentedQueenAntColony:
    def __init__(self, adj_matrix, feature_matrix, num_communities, device='cuda', 
                 aug_weight=0.5, top_k=10):  # <--- REGLAGE EQULIBRÉ
        """
        FE-ACO ULTIMATE ENGINE (Version Gold Standard)
        C'est la configuration qui a battu Node2Vec (0.49 NMI).
        """
        self.device = device
        self.k = num_communities
        self.num_nodes = adj_matrix.shape[0]
        
        # --- 1. GRAPH AUGMENTATION ---
        feat_norm = F.normalize(feature_matrix, p=2, dim=1)
        sim_matrix = torch.mm(feat_norm, feat_norm.t())
        
        # Top-K à 10 pour avoir une vision plus large
        values, indices = torch.topk(sim_matrix, k=top_k, dim=1)
        sparse_sim = torch.zeros_like(sim_matrix)
        sparse_sim.scatter_(1, indices, values)
        
        # ATTENTION DYNAMIQUE
        degrees = adj_matrix.sum(dim=1).view(-1, 1)
        
        # On remet le boost à +2.0 (Suffisant pour k=10)
        dynamic_weight = aug_weight + (2.0 / (degrees + 1.0))
        
        weighted_semantics = dynamic_weight * sparse_sim
        
        self.fused_adj = adj_matrix + weighted_semantics
        self.fused_adj = self.fused_adj + torch.eye(self.num_nodes, device=device)
        
        deg = self.fused_adj.sum(dim=1).clamp(min=1.0)
        norm_fused_adj = deg.pow(-1).view(-1, 1) * self.fused_adj
        
        # Big Data Opt
        if self.num_nodes > 20000:
            self.norm_fused_adj = norm_fused_adj.to_sparse()
            self.is_sparse = True
        else:
            self.norm_fused_adj = norm_fused_adj
            self.is_sparse = False
            
        self._reset_queens()
            
    def _reset_queens(self):
        self.global_pheromone = torch.ones(self.num_nodes, self.k, device=self.device) / self.k
        perm = torch.randperm(self.num_nodes)
        queens = perm[:self.k]
        for i, node_idx in enumerate(queens):
            self.global_pheromone[node_idx, :] = 0.0
            self.global_pheromone[node_idx, i] = 1.0

    def step(self, current_round=0, total_rounds=80):
        """
        Recuit Quadratique Agressif (Pour gérer k=10).
        """
        progress = current_round / total_rounds
        
        # Inflation : On finit à 1.2 + 2.5 = 3.7
        # C'est nécessaire avec top_k=10 pour bien séparer les clusters à la fin
        inflation = 1.2 + (pow(progress, 2) * 2.5)
        
        # Évaporation : Amnésie finale (0.9)
        evaporation = 0.1 + (progress * 0.8) 
        
        # Propagation
        if self.is_sparse:
            neighbor_influence = torch.sparse.mm(self.norm_fused_adj, self.global_pheromone)
        else:
            neighbor_influence = torch.mm(self.norm_fused_adj, self.global_pheromone)
            
        probabilities = torch.pow(neighbor_influence, inflation)
        
        # Pruning
        prune_thresh = 0.01 + (progress * 0.05)
        probabilities[probabilities < prune_thresh] = 0.0
        
        row_sums = probabilities.sum(dim=1, keepdim=True)
        probabilities = torch.where(row_sums > 1e-9, probabilities / row_sums, 
                                    torch.ones_like(probabilities) / self.k)
        
        self.global_pheromone = (1 - evaporation) * self.global_pheromone + evaporation * probabilities
        
    def get_prediction(self):
        return torch.argmax(self.global_pheromone, dim=1)
    
    def get_confidence_vector(self):
        return self.global_pheromone