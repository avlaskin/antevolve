"""Initial program v4 to be evolved."""
import time
from abc import ABC, abstractmethod
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import networkx as nx
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_array

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
    Implements an ensemble link prediction model with advanced features and multiple classifiers.
    """
    def __init__(self):
        super().__init__()
        self.config = None
        self.ensemble = []  # List of (scaler, model) tuples
        self.feature_count = 0
        self.meta_model = None

    def train(self, config,
              train_labels: list,
              train_edg: list,
              real_nodes: list):
        st = time.monotonic()
        self.config = config
        self.nodes = set(real_nodes)
        self.train_edg = train_edg
        self.train_labels = train_labels
        # Build graph from training edges
        self.G = nx.Graph() if not self.config.isDirected() else nx.DiGraph()
        self.G.add_nodes_from(real_nodes)
        self.G.add_edges_from([edge for edge, label in zip(train_edg, train_labels) if label == 1])
        print('Training started. Getting pagerank.')
        # Compute PageRank as additional feature
        self.pagerank = nx.pagerank(self.G, alpha=0.85)
        
        # Compute betweenness centrality
        print('Nodes: ', len(real_nodes))
        print('Getting centrality. ', time.monotonic() - st)
        try:
            self.betweenness = nx.betweenness_centrality(self.G, myk=10000)
        except:
            self.betweenness = {node: 0.0 for node in self.G.nodes()}

        if len(real_nodes) < 30_000:
        # Compute closeness centrality
            print('Getting between centrality took.', time.monotonic() - st)
            try:
                self.closeness = nx.closeness_centrality(self.G)
            except:
                self.closeness = {node: 0.0 for node in self.G.nodes()}

            print('Getting closeness centrality took', time.monotonic() - st)
            # Compute eigenvector centrality. This op takes ennormous time.

            try:
                self.eigenvector = nx.eigenvector_centrality(self.G, max_iter=100, tol=1e-3)
            except:
                self.eigenvector = {node: 0.0 for node in self.G.nodes()}
        else:
            self.eigenvector = {node: 0.0 for node in self.G.nodes()}
            self.closeness = {node: 0.0 for node in self.G.nodes()}

        # Compute degree centrality
        print('Getting eug centrality.', time.monotonic() - st)
        self.degree_centrality = nx.degree_centrality(self.G)

        print('Getting SVD. ', time.monotonic() - st)
        # Build adjacency matrix for SVD and Katz index
        nodes_sorted = sorted(real_nodes)
        node_to_index = {node: idx for idx, node in enumerate(nodes_sorted)}
        n = len(nodes_sorted)
        rows = []
        cols = []
        data = []
        for idx, edge in enumerate(train_edg):
            if train_labels[idx] == 1:
                u, v = edge
                i, j = node_to_index[u], node_to_index[v]
                rows.append(i)
                cols.append(j)
                data.append(1)
                if not self.config.isDirected():
                    rows.append(j)
                    cols.append(i)
                    data.append(1)
        
        adj_matrix = csr_array((data, (rows, cols)), shape=(n, n))
        
        try:
            U, S, Vt = svds(adj_matrix.astype(float), k=min(16, n-1))
            embeddings = U * np.sqrt(S)
        except Exception as e:
            embeddings = np.random.normal(0, 0.01, (n, 16))
        
        self.node_to_index = node_to_index
        self.embeddings = embeddings
        self.adj_matrix = adj_matrix
        
        # Compute matrix powers for Katz index
        A_sq = adj_matrix @ adj_matrix
        A_cu = adj_matrix @ A_sq
        A_qu = A_cu @ adj_matrix  # 4th power
        self.A_sq = A_sq
        self.A_cu = A_cu
        self.A_qu = A_qu
        
        if self.config.isDirected():
            self.clustering_coeffs = {node: 0.0 for node in self.G.nodes()}
        else:
            self.clustering_coeffs = nx.clustering(self.G)
        print('Other things.', time.monotonic() - st)
        # Compute features for training edges
        features = []
        for (u, v), label in zip(train_edg, train_labels):
            # Common neighbors
            try:
                common_neighbors = list(nx.common_neighbors(self.G, u, v))
            except:
                common_neighbors = []
            cn = len(common_neighbors)
            
            # Adamic-Adar
            aa = 0.0
            for n in common_neighbors:
                deg = self.G.degree(n)
                if deg > 1:
                    aa += 1.0 / np.log(deg)
            
            # Jaccard coefficient
            neighbors_u = set(self.G.neighbors(u))
            neighbors_v = set(self.G.neighbors(v))
            union_size = len(neighbors_u | neighbors_v)
            jaccard = cn / union_size if union_size > 0 else 0.0
            
            # Resource allocation
            ra = 0.0
            for n in common_neighbors:
                deg = self.G.degree(n)
                ra += 1.0 / deg if deg > 0 else 0.0
            
            # Preferential attachment
            deg_u = self.G.degree(u)
            deg_v = self.G.degree(v)
            pa = deg_u * deg_v
            
            # Cosine similarity (neighborhood)
            cosine = cn / np.sqrt(deg_u * deg_v) if deg_u > 0 and deg_v > 0 else 0.0
            
            # Embedding similarity from SVD
            i, j = self.node_to_index.get(u, 0), self.node_to_index.get(v, 0)
            if u in self.node_to_index and v in self.node_to_index:
                emb_u = self.embeddings[i]
                emb_v = self.embeddings[j]
                emb_sim = np.dot(emb_u, emb_v) / (np.linalg.norm(emb_u) * np.linalg.norm(emb_v) + 1e-8)
            else:
                emb_sim = 0.0
            
            # Shortest path length
            try:
                sp = nx.shortest_path_length(self.G, u, v)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                sp = -1
            
            # Triangle counts (only for undirected graphs)
            if self.config.isDirected():
                tri_u = 0
                tri_v = 0
            else:
                tri_u = nx.triangles(self.G, u) if u in self.G else 0
                tri_v = nx.triangles(self.G, v) if v in self.G else 0
            
            clus_u = self.clustering_coeffs.get(u, 0)
            clus_v = self.clustering_coeffs.get(v, 0)
            
            # Katz index (paths of length 2, 3, and 4)
            katz_2 = self.A_sq[i, j] if u in self.node_to_index and v in self.node_to_index else 0
            katz_3 = self.A_cu[i, j] if u in self.node_to_index and v in self.node_to_index else 0
            katz_4 = self.A_qu[i, j] if u in self.node_to_index and v in self.node_to_index else 0
            
            # Total neighbors
            total_neighbors = len(neighbors_u | neighbors_v)
            
            # Second order proximity (common neighbors of neighbors)
            shared_neighbor_connections = 0
            if u in self.G and v in self.G:
                for n1 in common_neighbors:
                    for n2 in common_neighbors:
                        if n1 != n2 and self.G.has_edge(n1, n2):
                            shared_neighbor_connections += 1
            
            # Centrality features
            pr_u = self.pagerank.get(u, 0)
            pr_v = self.pagerank.get(v, 0)
            bt_u = self.betweenness.get(u, 0)
            bt_v = self.betweenness.get(v, 0)
            cl_u = self.closeness.get(u, 0)
            cl_v = self.closeness.get(v, 0)
            ev_u = self.eigenvector.get(u, 0)
            ev_v = self.eigenvector.get(v, 0)
            dc_u = self.degree_centrality.get(u, 0)
            dc_v = self.degree_centrality.get(v, 0)
            
            # Hub promoted index
            hpi = cn / max(deg_u, deg_v) if max(deg_u, deg_v) > 0 else 0.0
            
            # Hub depressed index
            hdi = cn / min(deg_u, deg_v) if min(deg_u, deg_v) > 0 else 0.0
            
            # Leicht-Holme-Newman index
            lhn = cn / (deg_u * deg_v) if deg_u > 0 and deg_v > 0 else 0.0
            
            # Salton index
            salton = cn / np.sqrt(deg_u * deg_v) if deg_u > 0 and deg_v > 0 else 0.0
            
            # Sørensen index
            sorensen = 2 * cn / (deg_u + deg_v) if (deg_u + deg_v) > 0 else 0.0
            
            # Average neighbor degree
            avg_deg_u = np.mean([self.G.degree(n) for n in self.G.neighbors(u)]) if self.G.degree(u) > 0 else 0
            avg_deg_v = np.mean([self.G.degree(n) for n in self.G.neighbors(v)]) if self.G.degree(v) > 0 else 0
            
            # Variance of neighbor degrees
            var_deg_u = np.var([self.G.degree(n) for n in self.G.neighbors(u)]) if self.G.degree(u) > 0 else 0
            var_deg_v = np.var([self.G.degree(n) for n in self.G.neighbors(v)]) if self.G.degree(v) > 0 else 0
            
            # Node ID based features (assuming temporal ordering)
            node_id_diff = abs(u - v)
            node_id_ratio = min(u, v) / (max(u, v) + 1e-8)
            
            # Enhanced clustering coefficient features
            avg_clustering = np.mean([self.clustering_coeffs.get(n, 0) for n in common_neighbors]) if cn > 0 else 0
            
            features.append([
                aa, jaccard, ra, cn, pa, cosine, emb_sim, clus_u, clus_v, 
                deg_u, deg_v, pr_u, pr_v, bt_u, bt_v, cl_u, cl_v,
                sp, tri_u, tri_v, katz_2, katz_3, katz_4, total_neighbors, 
                shared_neighbor_connections, hpi, hdi, lhn, salton, sorensen,
                ev_u, ev_v, dc_u, dc_v, avg_deg_u, avg_deg_v, var_deg_u, var_deg_v,
                node_id_diff, node_id_ratio, avg_clustering
            ])
        print('Learning things.', time.monotonic() - st)        
        # Train ensemble with multiple models and scaling for stacking
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        models = [
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42, max_depth=12, min_samples_split=5)),
            ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42, learning_rate=0.1, max_depth=6)),
            ('lr1', LogisticRegression(random_state=42, max_iter=1000, C=0.5)),
            ('lr2', LogisticRegression(random_state=42, max_iter=1000, C=1.0, penalty='l1', solver='liblinear')),
            ('rf2', RandomForestClassifier(n_estimators=150, random_state=43, max_depth=10, min_samples_split=3, min_samples_leaf=2))
        ]
        
        self.ensemble = []  # will store (scaler, model)
        for name, model in models:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(np.array(features))
            model.fit(X_scaled, train_labels)
            self.ensemble.append((scaler, model))
        
        # Compute meta-features for stacking with weighted averaging
        X_meta = []
        model_scores = []
        for scaler, model in self.ensemble:
            X_scaled = scaler.transform(np.array(features))
            pred = model.predict_proba(X_scaled)[:, 1]
            X_meta.append(pred)
            # Compute model performance on training set for weighting
            train_pred = model.predict(X_scaled)
            from sklearn.metrics import accuracy_score
            model_scores.append(accuracy_score(train_labels, train_pred))
        X_meta = np.array(X_meta).T
        
        # Train meta-model with better regularization and weighted features
        self.meta_model = LogisticRegression(max_iter=2000, random_state=42, C=0.8)
        # Add sample weights based on model performance
        weights = np.array(model_scores) / np.sum(model_scores)
        X_meta_weighted = X_meta * weights
        self.meta_model.fit(X_meta_weighted, train_labels)
        
        self.feature_count = len(features[0]) if features else 0

    def predict(self, test_edges: list[tuple[int, int]]) -> np.array:
        feature_vectors = []
        for u, v in test_edges:
            # Check if nodes are present in the graph
            if u not in self.G or v not in self.G:
                feature_vector = [0.0] * self.feature_count
            else:
                # Common neighbors
                try:
                    common_neighbors = list(nx.common_neighbors(self.G, u, v))
                except:
                    common_neighbors = []
                cn = len(common_neighbors)
                
                # Adamic-Adar
                aa = 0.0
                for n in common_neighbors:
                    deg = self.G.degree(n)
                    if deg > 1:
                        aa += 1.0 / np.log(deg)
                
                # Jaccard coefficient
                neighbors_u = set(self.G.neighbors(u))
                neighbors_v = set(self.G.neighbors(v))
                union_size = len(neighbors_u | neighbors_v)
                jaccard = cn / union_size if union_size > 0 else 0.0
                
                # Resource allocation
                ra = 0.0
                for n in common_neighbors:
                    deg = self.G.degree(n)
                    ra += 1.0 / deg if deg > 0 else 0.0
                
                # Preferential attachment
                deg_u = self.G.degree(u)
                deg_v = self.G.degree(v)
                pa = deg_u * deg_v
                
                # Cosine similarity (neighborhood)
                cosine = cn / np.sqrt(deg_u * deg_v) if deg_u > 0 and deg_v > 0 else 0.0
                
                # Embedding similarity from SVD
                i, j = self.node_to_index.get(u, 0), self.node_to_index.get(v, 0)
                if u in self.node_to_index and v in self.node_to_index:
                    emb_u = self.embeddings[i]
                    emb_v = self.embeddings[j]
                    emb_sim = np.dot(emb_u, emb_v) / (np.linalg.norm(emb_u) * np.linalg.norm(emb_v) + 1e-8)
                else:
                    emb_sim = 0.0
                
                # Shortest path length
                try:
                    sp = nx.shortest_path_length(self.G, u, v)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    sp = -1
                
                # Triangle counts (only for undirected graphs)
                if self.config.isDirected():
                    tri_u = 0
                    tri_v = 0
                else:
                    tri_u = nx.triangles(self.G, u) if u in self.G else 0
                    tri_v = nx.triangles(self.G, v) if v in self.G else 0
                
                clus_u = self.clustering_coeffs.get(u, 0)
                clus_v = self.clustering_coeffs.get(v, 0)
                
                # Katz index (paths of length 2, 3, and 4)
                katz_2 = self.A_sq[i, j] if u in self.node_to_index and v in self.node_to_index else 0
                katz_3 = self.A_cu[i, j] if u in self.node_to_index and v in self.node_to_index else 0
                katz_4 = self.A_qu[i, j] if u in self.node_to_index and v in self.node_to_index else 0
                
                # Total neighbors
                total_neighbors = len(neighbors_u | neighbors_v)
                
                # Second order proximity (common neighbors of neighbors)
                shared_neighbor_connections = 0
                if u in self.G and v in self.G:
                    for n1 in common_neighbors:
                        for n2 in common_neighbors:
                            if n1 != n2 and self.G.has_edge(n1, n2):
                                shared_neighbor_connections += 1
                
                # Centrality features
                pr_u = self.pagerank.get(u, 0)
                pr_v = self.pagerank.get(v, 0)
                bt_u = self.betweenness.get(u, 0)
                bt_v = self.betweenness.get(v, 0)
                cl_u = self.closeness.get(u, 0)
                cl_v = self.closeness.get(v, 0)
                ev_u = self.eigenvector.get(u, 0)
                ev_v = self.eigenvector.get(v, 0)
                dc_u = self.degree_centrality.get(u, 0)
                dc_v = self.degree_centrality.get(v, 0)
                
                # Hub promoted index
                hpi = cn / max(deg_u, deg_v) if max(deg_u, deg_v) > 0 else 0.0
                
                # Hub depressed index
                hdi = cn / min(deg_u, deg_v) if min(deg_u, deg_v) > 0 else 0.0
                
                # Leicht-Holme-Newman index
                lhn = cn / (deg_u * deg_v) if deg_u > 0 and deg_v > 0 else 0.0
                
                # Salton index
                salton = cn / np.sqrt(deg_u * deg_v) if deg_u > 0 and deg_v > 0 else 0.0
                
                # Sørensen index
                sorensen = 2 * cn / (deg_u + deg_v) if (deg_u + deg_v) > 0 else 0.0
                
                # Average neighbor degree
                avg_deg_u = np.mean([self.G.degree(n) for n in self.G.neighbors(u)]) if self.G.degree(u) > 0 else 0
                avg_deg_v = np.mean([self.G.degree(n) for n in self.G.neighbors(v)]) if self.G.degree(v) > 0 else 0
                
                # Variance of neighbor degrees
                var_deg_u = np.var([self.G.degree(n) for n in self.G.neighbors(u)]) if self.G.degree(u) > 0 else 0
                var_deg_v = np.var([self.G.degree(n) for n in self.G.neighbors(v)]) if self.G.degree(v) > 0 else 0
                
                # Node ID based features (assuming temporal ordering)
                node_id_diff = abs(u - v)
                node_id_ratio = min(u, v) / (max(u, v) + 1e-8)
                
                # Enhanced clustering coefficient features
                avg_clustering = np.mean([self.clustering_coeffs.get(n, 0) for n in common_neighbors]) if cn > 0 else 0
                
                feature_vector = [
                    aa, jaccard, ra, cn, pa, cosine, emb_sim, clus_u, clus_v, 
                    deg_u, deg_v, pr_u, pr_v, bt_u, bt_v, cl_u, cl_v,
                    sp, tri_u, tri_v, katz_2, katz_3, katz_4, total_neighbors, 
                    shared_neighbor_connections, hpi, hdi, lhn, salton, sorensen,
                    ev_u, ev_v, dc_u, dc_v, avg_deg_u, avg_deg_v, var_deg_u, var_deg_v,
                    node_id_diff, node_id_ratio, avg_clustering
                ]
            
            feature_vectors.append(feature_vector)
        
        X_test = np.array(feature_vectors)
        X_meta_test = []
        for scaler, model in self.ensemble:
            X_test_scaled = scaler.transform(X_test)
            pred = model.predict_proba(X_test_scaled)[:, 1]
            X_meta_test.append(pred)
        X_meta_test = np.array(X_meta_test).T
        
        meta_probs = self.meta_model.predict_proba(X_meta_test)
        return meta_probs

# START DO NOT MODIFY

def config_factory() -> Configuration:
    """Creates a config for the model."""
    return MySimpleConfig(directed=False)


def solver_factory() -> MyLinkPredictionMethod:
    """Main solver factory."""
    solver = MyLinkPredictionMethod()
    return solver

# END DO NOT MODIFY
