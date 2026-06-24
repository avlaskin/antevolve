"""Link predicting algorithm evolved using Gemini,
1000 iterations. Using only synthetic graphs for training.
"""

from abc import ABC, abstractmethod
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

import sklearn
import networkx as nx
import numpy as np
import math

# START DO NOT MODIFY

class Configuration(ABC):
    """Model config abstract class.""" 
    @abstractmethod
    def isDirected(self):
        pass

class AbsModel(ABC):
    """Model abstract class."""
    @abstractmethod
    def train(self,
              config: Configuration, 
              train_labels: list[int], 
              train_edg: list[tuple[int, int]],
              real_nodes: list[int]):
        pass
    
    @abstractmethod
    def predict(self, test_edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Main method that predicts the """
        pass
    
    def compute_auc(self, predicted: list, test_labels: list):
        """Computes prediction score."""
        assert len(predicted.shape) > 1 and len(test_labels) == predicted.shape[0]
        #pr = [0.0 if p[0] > p[1] else 1.0 for p in predicted]
        pr = [p[1] for p in predicted]
        res = metrics.roc_auc_score(test_labels, pr)
        return res
    
    def reset(self):
        """Resets the class instance."""
        self.config = None

### END OF DO NOT MODIFY 


class MySimpleConfig(Configuration):
    """ Simplest config for heuristics models."""
    def __init__(self, directed: bool):
        self.directed = directed
       
    def isDirected(self):
        return self.directed


class MyLinkPredictionMethod(AbsModel):
    """
    Implements a link prediction model.
    """
    def __init__(self):
        super().__init__()
        self.config = None # will set it later

    def _extract_features(self, G, edges, real_nodes):
        """Extracts a 40-dimensional hybrid topological-spectral feature vector."""
        is_dir = G.is_directed()
        H = G.to_undirected() if is_dir else G
        nodes_in_h = set(H.nodes())
        node_to_idx = {node: i for i, node in enumerate(real_nodes)}
        degrees = dict(H.degree())
        avg_deg = sum(degrees.values()) / len(degrees) if degrees else 1.0
        pr_map = nx.pagerank(H, alpha=0.85) if H.number_of_edges() > 0 else {}
        clust_map = nx.clustering(H)
        embeddings = None
        if H.number_of_edges() > 10:
            adj = nx.adjacency_matrix(H, nodelist=real_nodes).astype(float)
            k = min(32, len(real_nodes) - 2)
            try:
                u, s, _ = svds(adj, k=k)
                embeddings = u * np.sqrt(s)
            except: embeddings = None
        valid_edges = [(u, v) for u, v in edges if u in nodes_in_h and v in nodes_in_h]
        def to_map(gen): return {(u,v): val for u,v,val in gen}
        jc_map, aa_map = to_map(nx.jaccard_coefficient(H, valid_edges)), to_map(nx.adamic_adar_index(H, valid_edges))
        pa_map, ra_map = to_map(nx.preferential_attachment(H, valid_edges)), to_map(nx.resource_allocation_index(H, valid_edges))
        features = []
        for u, v in edges:
            if u not in nodes_in_h or v not in nodes_in_h:
                features.append([0.0] * 43); continue
            
            d_u, d_v = degrees.get(u, 0), degrees.get(v, 0)
            u_neigh, v_neigh = set(H[u]), set(H[v])
            cn_list = u_neigh.intersection(v_neigh)
            cn = len(cn_list)
            
            # L3 Normalized Path Score
            l3 = 0
            if d_u > 0 and d_v > 0:
                for w1 in u_neigh:
                    for w2 in v_neigh:
                        if H.has_edge(w1, w2):
                            l3 += 1.0 / math.sqrt(degrees.get(w1, 1) * degrees.get(w2, 1))

            # Spectral Similarity, Distance, and Katz Approximation
            spec_sim, spec_dist, spec_cos, katz_sim = 0.0, 0.0, 0.0, 0.0
            if embeddings is not None:
                u_emb, v_emb = embeddings[node_to_idx[u]], embeddings[node_to_idx[v]]
                spec_sim = np.dot(u_emb, v_emb)
                spec_dist = np.linalg.norm(u_emb - v_emb)
                norm_uv = (np.linalg.norm(u_emb) * np.linalg.norm(v_emb))
                spec_cos = spec_sim / (norm_uv + 1e-9)
                
                # Diffusion-based spectral proximity and multi-scale similarity
                # We use an exponential weighting to simulate a heat-kernel process
                s_diff = np.exp(s / (s.max() + 1e-9))
                katz_sim = np.sum(u_emb * v_emb * s_diff)
                spec_sim = np.sum(u_emb * v_emb * np.log1p(s))
            
            # Normalization variants
            salton = cn / (math.sqrt(d_u * d_v) + 1e-7)
            hpi = cn / min(d_u, d_v) if min(d_u, d_v) > 0 else 0
            hdi = cn / max(d_u, d_v) if max(d_u, d_v) > 0 else 0
            lhn = cn / (d_u * d_v + 1e-7)
            
            cn_edges = H.subgraph(cn_list).number_of_edges() if cn > 1 else 0
            fts = [
                cn, jc_map.get((u,v), 0), aa_map.get((u,v), 0), ra_map.get((u,v), 0), pa_map.get((u,v), 0),
                l3, spec_sim, spec_dist, spec_cos, salton, hpi, hdi, lhn,
                d_u + d_v, d_u * d_v, abs(d_u - d_v),
                pr_map.get(u,0), pr_map.get(v,0), pr_map.get(u,0) * pr_map.get(v,0),
                sum(pr_map.get(w,0) for w in cn_list),
                len(u_neigh.union(v_neigh)), cn / (d_u + d_v - cn + 1e-7),
                math.log1p(cn), math.log1p(d_u * d_v),
                int(G.has_edge(u,v)) if is_dir else 0, int(G.has_edge(v,u)) if is_dir else 0,
                clust_map.get(u,0), clust_map.get(v,0), d_u/avg_deg, d_v/avg_deg,
                min(d_u, d_v), max(d_u, d_v),
                np.mean(u_emb * v_emb) if embeddings is not None else 0,
                cn_edges, # Community density
                abs(pr_map.get(u,0) - pr_map.get(v,0)),
                spec_dist / (cn + 1.0), # Spectral-Structural interaction
                pr_map.get(u,0) / (clust_map.get(u,0) + 1e-5),
                pr_map.get(v,0) / (clust_map.get(v,0) + 1e-5),
                (clust_map.get(u,0) + clust_map.get(v,0)) / 2.0,
                (d_u * clust_map.get(u,0)) * (d_v * clust_map.get(v,0)),
                cn_edges / (cn + 1e-7), # Internal neighborhood density
                sum(clust_map.get(w, 0) for w in cn_list) if cn > 0 else 0,
                sum(degrees.get(w, 0) for w in cn_list) / (cn + 1e-7)
            ]
            features.append([float(f) for f in fts])
        return np.array(features)

    def train(self, config,
              train_labels: list,
              train_edg: list,
              real_nodes: list):
        """
        Uses a 3-Fold Out-of-Fold (OOF) feature extraction strategy.
        This maximizes the graph density used for training features while preventing leakage.
        """

        self.config = config
        self.real_nodes = real_nodes
        self.G = nx.DiGraph() if config.isDirected() else nx.Graph()
        self.G.add_nodes_from(real_nodes)
        pos_indices = np.array([i for i, label in enumerate(train_labels) if label == 1])
        neg_indices = np.array([i for i, label in enumerate(train_labels) if label == 0])
        self.G.add_edges_from([train_edg[i] for i in pos_indices])
        
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        X_train_list, y_train_list = [], []
        
        for t_idx, v_idx in kf.split(pos_indices):
            # Build graph using 2/3 of positive edges
            G_fold = nx.DiGraph() if config.isDirected() else nx.Graph()
            G_fold.add_nodes_from(real_nodes)
            G_fold.add_edges_from([train_edg[pos_indices[i]] for i in t_idx])
            
            # Extract features for the remaining 1/3 of positives and 1/3 of negatives
            # We shuffle negatives and take a proportional slice
            np.random.seed(42); np.random.shuffle(neg_indices)
            neg_slice = neg_indices[:len(neg_indices)//3]
            
            fold_edges = [train_edg[pos_indices[i]] for i in v_idx] + [train_edg[i] for i in neg_slice]
            fold_labels = [1] * len(v_idx) + [0] * len(neg_slice)
            
            X_fold = self._extract_features(G_fold, fold_edges, real_nodes)
            X_train_list.append(X_fold)
            y_train_list.extend(fold_labels)
            
        X_train = np.vstack(X_train_list)
        y_train = np.array(y_train_list)
        
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.clf = HistGradientBoostingClassifier(
            max_iter=1500, learning_rate=0.04,
            max_depth=10, l2_regularization=5.0, 
            min_samples_leaf=20, class_weight='balanced',
            random_state=42, early_stopping=True,
            n_iter_no_change=50, categorical_features=[24, 25]
        )
        self.clf.fit(X_train, y_train)

    def predict(self, test_edges: list[tuple[int, int]]) -> np.array:
        """Predicts edge probabilities using the trained Meta-Learner."""
        if not hasattr(self, 'clf'):
            return np.array([(0.5, 0.5)] * len(test_edges))
        X_test = self._extract_features(self.G, test_edges, self.real_nodes)
        return self.clf.predict_proba(X_test)

# START DO NOT MODIFY

def config_factory() -> Configuration:
    """Creates a config for the model."""
    return MySimpleConfig(directed=False)


def solver_factory() -> MyLinkPredictionMethod:
    """Main solver factory."""
    solver = MyLinkPredictionMethod()
    return solver

# END DO NOT MODIFY
