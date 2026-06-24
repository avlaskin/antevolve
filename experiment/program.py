"""Initial program for link prediction."""

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


class MyLinkPredictionMethod(AbsModel):
    """
    Implements a link prediction model.
    """
    def __init__(self):
        super().__init__()
        self.config = None # will set it later

    def train(self, config,
              train_labels: list,
              train_edg: list,
              real_nodes: list):
        """Main method to implement by the LLM."""
        self.config = config
        self.nodes = set(real_nodes)
        self.train_edg = train_edg
        self.train_labels = train_labels
        # TODO: IMPLEMENT THE TRAINING HERE

    def predict(self, test_edges: list[tuple[int, int]]) -> np.array:
        """Main method that predicts the edge probability.
        Returns:
          numpy array that has list of tuples as per class probabilities of 
            the edge. Type hint: np.array(list[tuple[float, float]])
        """
        # TODO: IMPLEMENT THE EDGE PROBABILITY PREDICTION HERE
        # below is just a dummy code:
        edge_probs = []
        for _ in test_edges:
            # First float probability of non link, second probability of a link
            probability_of_link = 1.0 / len(test_edges)
            edge_probs.append((1.0 - probability_of_link, probability_of_link))
        return np.array(edge_probs)

# START DO NOT MODIFY

def config_factory() -> Configuration:
    """Creates a config for the model."""
    return MySimpleConfig(directed=False)


def solver_factory() -> MyLinkPredictionMethod:
    """Main solver factory."""
    solver = MyLinkPredictionMethod()
    return solver

# END DO NOT MODIFY
