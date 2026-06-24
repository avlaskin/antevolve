"""Initial program qwen-coder v1 to be evolved."""

from abc import ABC, abstractmethod
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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


class MyLinkPredictionMethod(AbsModel):
    """
    Implements a link prediction model with standard topological features and RandomForest.
    """
    def __init__(self):
        super().__init__()
        self.config = None
        self.scaler = StandardScaler()
        self.models = []
        self.graph = None

    def train(self, config,
              train_labels: list,
              train_edg: list,
              real_nodes: list):
        self.config = config
        self.nodes = set(real_nodes)
        self.train_edg = train_edg
        self.train_labels = train_labels
        # Build graph
        self.graph = nx.DiGraph() if self.config.isDirected() else nx.Graph()
        self.graph.add_nodes_from(real_nodes)
        self.graph.add_edges_from(train_edg)
        # Compute features
        # Create ensemble of models
        X = self._compute_features(train_edg)
        X_scaled = self.scaler.fit_transform(X)
        
        # Model 1: Random Forest (enhanced)
        rf_model = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=20, min_samples_split=5)
        rf_model.fit(X_scaled, train_labels)
        self.models.append(rf_model)
        
        # Model 2: Gradient Boosting
        from sklearn.ensemble import GradientBoostingClassifier
        gb_model = GradientBoostingClassifier(n_estimators=200, random_state=42, max_depth=8)
        gb_model.fit(X_scaled, train_labels)
        self.models.append(gb_model)
        
        # Model 3: Logistic Regression
        from sklearn.linear_model import LogisticRegression
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_scaled, train_labels)
        self.models.append(lr_model)

    def predict(self, test_edges: list[tuple[int, int]]) -> np.array:
        # Compute features for test edges
        X_test = self._compute_features(test_edges)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Ensemble prediction - average probabilities from all models
        probabilities = []
        for model in self.models:
            prob = model.predict_proba(X_test_scaled)
            probabilities.append(prob)
        
        # Average the probabilities from all models
        avg_probabilities = np.mean(probabilities, axis=0)
        return avg_probabilities

    def _compute_features(self, edges):
        features = []
        graph_undirected = self.graph if not self.config.isDirected() else self.graph.to_undirected()
        for u, v in edges:
            if u not in graph_undirected or v not in graph_undirected:
                features.append([0.0] * 20)
                continue
            neighbors_u = set(graph_undirected[u])
            neighbors_v = set(graph_undirected[v])
            common_nbrs = neighbors_u & neighbors_v
            common_count = len(common_nbrs)
            union_size = len(neighbors_u | neighbors_v)
            jaccard = common_count / union_size if union_size > 0 else 0.0
            adamic_adar = 0.0
            for w in common_nbrs:
                degree_w = len(graph_undirected[w])
                if degree_w > 1:
                    adamic_adar += 1.0 / np.log(degree_w)
            pref_attach = len(neighbors_u) * len(neighbors_v)
            resource_alloc = 0.0
            for w in common_nbrs:
                degree_w = len(graph_undirected[w])
                if degree_w > 0:
                    resource_alloc += 1.0 / degree_w
            cluster_u = nx.clustering(graph_undirected, u)
            cluster_v = nx.clustering(graph_undirected, v)
            degree_u = len(neighbors_u)
            degree_v = len(neighbors_v)
            node_diff = np.abs(u - v)
            node_prod = u * v
            total_degree = degree_u + degree_v
            geo_mean = np.sqrt(degree_u * degree_v) if degree_u * degree_v >= 0 else 0.0
            common_binary = 1.0 if common_count > 0 else 0.0
            ratio_u = degree_u / (total_degree + 1e-5)
            ratio_v = degree_v / (total_degree + 1e-5)
            log_deg_u = np.log(degree_u + 1)
            log_deg_v = np.log(degree_v + 1)
            # Quantum-inspired similarity based on node degrees
            quantum_sim = np.cos(degree_u - degree_v) * np.exp(-abs(degree_u - degree_v) / max(degree_u + degree_v, 1))
            
            # Temporal evolution similarity 
            temp_sim = np.sin((u - v) * np.pi / max(u + v, 1)) * np.log(max(u, v) - min(u, v) + 1)
            
            # Harmonic mean of clustering coefficients
            harmonic_cluster = 2 * cluster_u * cluster_v / max(cluster_u + cluster_v, 1e-8)
            
            normalized_total_degree = total_degree / (len(graph_undirected) + 1e-5)
            features.append([
                common_count,
                jaccard,
                adamic_adar,
                pref_attach,
                resource_alloc,
                cluster_u,
                cluster_v,
                degree_u,
                degree_v,
                node_diff,
                node_prod,
                common_count * jaccard,
                total_degree,
                geo_mean,
                common_binary,
                ratio_u,
                ratio_v,
                log_deg_u,
                log_deg_v,
                normalized_total_degree,
                quantum_sim,
                temp_sim,
                harmonic_cluster
            ])
        return np.array(features)

# START DO NOT MODIFY

def config_factory() -> Configuration:
    """Creates a config for the model."""
    return MySimpleConfig(directed=False)


def solver_factory() -> MyLinkPredictionMethod:
    """Main solver factory."""
    solver = MyLinkPredictionMethod()
    return solver

# END DO NOT MODIFY