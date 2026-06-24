"""Initial program v4 to be evolved."""

from abc import ABC, abstractmethod
from sklearn import metrics

import sklearn
import sklearn.ensemble
import networkx as nx
import numpy as np
import zlib
import cmath
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
    "Quantum Chromo-Dynamic Entanglement with Algorithmic Information Torsion" (QCDE-AIT).
    Models links as particle interactions in a Hilbert space using Kolmogorov complexity
    approximations (zlib NCD) and quantum phase interference patterns.
    """
    def __init__(self):
        super().__init__()
        self.config = None
        self.G = None
        self.wl_labels = {}
        self.deg_map = {}
        self.triangles_map = {}
        self.max_id = 1
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        self.clf = sklearn.ensemble.HistGradientBoostingClassifier(
            max_iter=1200,
            max_depth=25,
            learning_rate=0.04,
            l2_regularization=2.0,
            class_weight='balanced',
            random_state=1337
        )

    def _get_algorithmic_torsion(self, u_nb, v_nb):
        """Approximates Kolmogorov Complexity similarity using NCD."""
        s_u = ",".join(map(str, sorted(list(u_nb)))).encode()
        s_v = ",".join(map(str, sorted(list(v_nb)))).encode()
        c_u = len(zlib.compress(s_u))
        c_v = len(zlib.compress(s_v))
        c_uv = len(zlib.compress(s_u + b"|" + s_v))
        ncd = (c_uv - min(c_u, c_v)) / max(c_u, c_v)
        return ncd

    def _get_quantum_interference(self, u, v, du, dv):
        """Calculates constructive/destructive interference between node phases."""
        # Phase derived from ID and local density
        phi_u = (u * math.pi / (self.max_id + 1)) + (du * 0.1)
        phi_v = (v * math.pi / (self.max_id + 1)) + (dv * 0.1)
        psi_u = cmath.exp(1j * phi_u)
        psi_v = cmath.exp(1j * phi_v)
        # Probability density of the 'entangled' state
        interference = abs(psi_u + psi_v)**2
        return interference

    def _extract_features(self, u, v, is_train=False):
        if u not in self.G or v not in self.G:
            return np.zeros(58)

        u_nb = set(self.G.neighbors(u))
        v_nb = set(self.G.neighbors(v))
        if is_train:
            u_nb.discard(v); v_nb.discard(u)
        
        du, dv = len(u_nb), len(v_nb)
        common = u_nb & v_nb
        cn = len(common)
        
        # 1. Quantum & Information Metrics
        ncd = self._get_algorithmic_torsion(u_nb, v_nb)
        q_interf = self._get_quantum_interference(u, v, du, dv)
        
        # 2. Chromodynamic Baryon Flux (Triangle-based)
        tu, tv = self.triangles_map.get(u, 0), self.triangles_map.get(v, 0)
        baryon_density = (tu + tv) / (du + dv + 1e-9)
        gluon_flux = cn * (tu + 1) * (tv + 1)

        # 3. Bit-wise Genetic Torsion
        xor_id = u ^ v
        hamming = bin(xor_id).count('1')
        interlock = bin(u & v).count('1')
        gray_u = u ^ (u >> 1)
        gray_v = v ^ (v >> 1)
        gray_dist = bin(gray_u ^ gray_v).count('1')

        # 4. WL-Manifold Signature (Structural DNA)
        u_wl = self.wl_labels.get(u, ["0"])
        v_wl = self.wl_labels.get(v, ["0"])
        wl_sim = sum(1 for a, b in zip(u_wl, v_wl) if a == b)

        # 5. Topological Curvature & Traditional Core
        aa = sum(1.0 / np.log(self.deg_map.get(w, 2) + 1.01) for w in common)
        ra = sum(1.0 / (self.deg_map.get(w, 1) + 1e-9) for w in common)
        jaccard = cn / (len(u_nb | v_nb) + 1e-9)
        preferential = du * dv
        
        # 6. Harmonic Resonance
        u_p = np.array([u % p for p in self.primes])
        v_p = np.array([v % p for p in self.primes])
        prime_resonance = np.sum(u_p == v_p)
        gcd_val = math.gcd(u, v)

        return np.array([
            cn, aa, ra, jaccard, preferential,
            ncd, q_interf, baryon_density, gluon_flux,
            hamming, interlock, gray_dist, wl_sim,
            tu, tv, np.sqrt(tu * tv + 1e-9),
            abs(du - dv), np.log1p(du), np.log1p(dv),
            # Parity and Spectral residues
            float(u % 2 == v % 2), float(u % 3 == v % 3),
            u % 7, v % 7, u % 13, v % 13, u % 17, v % 17,
            # Local Clustering Coefficients
            (2*tu)/(du*(du-1)+1e-9) if du > 1 else 0,
            (2*tv)/(dv*(dv-1)+1e-9) if dv > 1 else 0,
            # ID Scaling
            u / self.max_id, v / self.max_id,
            abs(u - v) / self.max_id,
            # Neighborhood Entropy
            np.mean([self.deg_map.get(w, 0) for w in common]) if cn > 0 else 0,
            # Advanced Bitwise
            (u << 1) ^ v, (v << 1) ^ u,
            bin(u).count('1') / (bin(v).count('1') + 1e-9),
            # Path Contraction
            cn / (du + dv - cn + 1e-9),
            # Geometric mean of degrees
            np.sqrt(du * dv),
            # Harmonic mean of degrees
            2.0 / (1.0/(du+1) + 1.0/(dv+1)),
            # Symmetry ratio
            min(du, dv) / (max(du, dv) + 1e-9),
            # Prime Resonance Metrics
            prime_resonance, gcd_val,
            # Manifold projections
            math.sin(u), math.cos(v),
            # Topological "Heat"
            cn / (np.sqrt(tu + tv + 1)),
            # Entropy proxies
            len(set(u_p)), len(set(v_p)),
            # Bitwise rotations
            (u >> 2) | (v << 2),
            # Higher order residues
            u % 31, v % 31, u % 64, v % 64,
            # Chaos factor
            float(hash(f"{u}:{v}") % 1000) / 1000.0,
            # Adjacency Density
            (cn + 1) / (min(du, dv) + 1),
            # Potential Well
            (du + dv) / (abs(u - v) + 1e-1)
        ])

    def train(self, config, train_labels, train_edg, real_nodes):
        self.config = config
        self.max_id = max(real_nodes) if real_nodes else 1
        self.G = nx.Graph() if not config.isDirected() else nx.DiGraph()
        self.G.add_nodes_from(real_nodes)
        for i, edge in enumerate(train_edg):
            if train_labels[i] == 1:
                self.G.add_edge(*edge)
        
        undir = self.G.to_undirected() if config.isDirected() else self.G
        self.deg_map = dict(undir.degree())
        self.triangles_map = nx.triangles(undir)
        
        # Compute Weisfeiler-Lehman Manifold Labels
        labels = {n: str(self.deg_map.get(n, 0)) for n in real_nodes}
        self.wl_labels = {n: [] for n in real_nodes}
        for _ in range(3): # Increased to 3 iterations for deeper structural DNA
            new_labels = {}
            for n in real_nodes:
                neigh_labels = sorted([labels[neigh] for neigh in undir.neighbors(n)])
                combined = labels[n] + "".join(neigh_labels)
                new_labels[n] = str(hash(combined))
                self.wl_labels[n].append(new_labels[n])
            labels = new_labels

        X = [self._extract_features(u, v, is_train=True) for u, v in train_edg]
        self.clf.fit(X, train_labels)

    def predict(self, test_edges: list[tuple[int, int]]) -> np.array:
        X = [self._extract_features(u, v, is_train=False) for u, v in test_edges]
        return self.clf.predict_proba(X)

# START DO NOT MODIFY

def config_factory() -> Configuration:
    """Creates a config for the model."""
    return MySimpleConfig(directed=False)


def solver_factory() -> MyLinkPredictionMethod:
    """Main solver factory."""
    solver = MyLinkPredictionMethod()
    return solver

# END DO NOT MODIFY
