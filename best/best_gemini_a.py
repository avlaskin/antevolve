"""Link predicting algorithm evolved using Gemini,
1000 iterations. Using only synthetic + four real networks
for training.
"""

from abc import ABC, abstractmethod
from sklearn import metrics

import sklearn
import networkx as nx
import numpy as np

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


from sklearn.ensemble import ExtraTreesClassifier

class MyLinkPredictionMethod(AbsModel):
    """
    Advanced Link Prediction using Multi-scale Path Resonance (MPR) 
    and Leakage-Corrected Local Core-Shell Synergy (LCSS).
    """
    def __init__(self):
        super().__init__()
        self.config = None
        self.graph = None
        self.degree_map = {}
        self.triangles_map = {}
        self.adj_map = {}
        self.neigh_deg_sum_map = {}
        
        # High-capacity ensemble to capture non-linear topological interactions
        self.clf = ExtraTreesClassifier(
            n_estimators=800,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced_subsample',
            n_jobs=-1
        )

    def _extract_features(self, u, v, is_positive_train=False):
        """Extracts 24 topological features with precise leakage correction."""
        if u not in self.adj_map or v not in self.adj_map:
            return np.zeros(24)

        # 1. Neighborhood Extraction & Leakage Adjustment
        u_nb = self.adj_map[u].copy()
        v_nb = self.adj_map[v].copy()
        
        d_u_orig = self.degree_map.get(u, 0)
        d_v_orig = self.degree_map.get(v, 0)
        
        if is_positive_train:
            u_nb.discard(v)
            v_nb.discard(u)

        d_u_eff, d_v_eff = len(u_nb), len(v_nb)
        common = u_nb & v_nb
        cn = len(common)
        union_size = len(u_nb | v_nb)
        
        # 2. Classic Heuristics
        jc = cn / union_size if union_size > 0 else 0
        pa = d_u_eff * d_v_eff
        log_pa = np.log1p(pa)
        sor = (2 * cn) / (d_u_eff + d_v_eff) if (d_u_eff + d_v_eff) > 0 else 0
        hpi = cn / min(d_u_eff, d_v_eff) if min(d_u_eff, d_v_eff) > 0 else 0
        hdi = cn / max(d_u_eff, d_v_eff) if max(d_u_eff, d_v_eff) > 0 else 0
        lhn = cn / (d_u_eff * d_v_eff) if pa > 0 else 0
        
        aa, ra = 0.0, 0.0
        for w in common:
            dw = self.degree_map.get(w, 1)
            if dw > 1:
                aa += 1.0 / np.log(dw)
                ra += 1.0 / dw

        # 3. Multi-scale Path Resonance (MPR) - Length 3 Paths
        mpr_weight = 0.0
        path3_count = 0
        if d_u_eff > 0 and d_v_eff > 0:
            # Iterate through smaller neighborhood for efficiency
            search_nb, target_nb = (u_nb, v_nb) if d_u_eff < d_v_eff else (v_nb, u_nb)
            for nbr in search_nb:
                nnbrs = self.adj_map.get(nbr, set())
                # Only look for paths u -> nbr -> target_nbr -> v
                overlap = nnbrs & target_nb
                if overlap:
                    path3_count += len(overlap)
                    deg_nbr = self.degree_map.get(nbr, 1)
                    for o_node in overlap:
                        deg_o = self.degree_map.get(o_node, 1)
                        mpr_weight += 1.0 / (np.log1p(deg_nbr) * np.log1p(deg_o))

        # 4. Local Core-Shell Synergy (LCSS)
        t_u = self.triangles_map.get(u, 0)
        t_v = self.triangles_map.get(v, 0)
        s_u = self.neigh_deg_sum_map.get(u, 0)
        s_v = self.neigh_deg_sum_map.get(v, 0)

        if is_positive_train:
            # Removing (u,v) removes 'cn' triangles from both ends
            t_u = max(0, t_u - cn)
            t_v = max(0, t_v - cn)
            # Neighbor degree sum adjustment
            s_u = max(0, s_u - d_v_orig)
            s_v = max(0, s_v - d_u_orig)

        cc_u = (2.0 * t_u) / (d_u_eff * (d_u_eff - 1)) if d_u_eff > 1 else 0
        cc_v = (2.0 * t_v) / (d_v_eff * (d_v_eff - 1)) if d_v_eff > 1 else 0
        and_u = s_u / d_u_eff if d_u_eff > 0 else 0
        and_v = s_v / d_v_eff if d_v_eff > 0 else 0

        return np.array([
            cn, jc, log_pa, aa, ra, sor, hpi, hdi, lhn,
            mpr_weight, path3_count, 
            cc_u, cc_v, cc_u * cc_v, (cc_u + cc_v) / 2.0,
            and_u, and_v, min(and_u, and_v), max(and_u, and_v),
            min(d_u_eff, d_v_eff), max(d_u_eff, d_v_eff),
            d_u_eff / (d_v_eff + 1e-6), t_u, t_v
        ])

    def train(self, config, train_labels, train_edg, real_nodes):
        self.config = config
        self.graph = nx.DiGraph() if config.isDirected() else nx.Graph()
        self.graph.add_nodes_from(real_nodes)
        
        for i, edge in enumerate(train_edg):
            if train_labels[i] == 1:
                self.graph.add_edge(*edge)
        
        # Precompute maps for O(1) feature extraction
        self.degree_map = dict(self.graph.degree())
        undir = self.graph.to_undirected() if config.isDirected() else self.graph
        self.triangles_map = nx.triangles(undir)
        self.adj_map = {n: set(self.graph.successors(n) if config.isDirected() else self.graph.neighbors(n)) 
                        for n in self.graph.nodes()}
        
        self.neigh_deg_sum_map = {n: sum(self.degree_map.get(nb, 0) for nb in nbrs) 
                                  for n, nbrs in self.adj_map.items()}

        X_train = [self._extract_features(e[0], e[1], is_positive_train=(train_labels[i] == 1)) 
                   for i, e in enumerate(train_edg)]
        self.clf.fit(X_train, train_labels)

    def predict(self, test_edges: list[tuple[int, int]]) -> np.array:
        X_test = [self._extract_features(e[0], e[1], is_positive_train=False) for e in test_edges]
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
