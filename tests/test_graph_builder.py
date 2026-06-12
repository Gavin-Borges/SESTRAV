import torch
import pytest
from src.gnn.graph_builder import GraphBuilder

def test_sequence_to_node_features():
    # Test valid sequence
    seq = "ACDEF"
    features = GraphBuilder.sequence_to_node_features(seq, max_len=11)
    
    assert features.shape == (11, 20) # max_len x vocab_size
    # A is index 0, C is index 1, D is index 2, E is index 3, F is index 4
    assert features[0, 0] == 1.0
    assert features[1, 1] == 1.0
    assert features[2, 2] == 1.0
    assert features[3, 3] == 1.0
    assert features[4, 4] == 1.0
    assert features[5].sum() == 0.0 # Padding

def test_build_chain_adj():
    adj = GraphBuilder.build_chain_adj(max_len=5)
    
    assert adj.shape == (5, 5)
    # Check symmetric normalized adj logic
    # Nodes 0 and 4 have degree 2 (self + 1 neighbor)
    # Nodes 1, 2, 3 have degree 3 (self + 2 neighbors)
    assert adj[0, 0] > 0
    assert adj[0, 1] > 0
    assert adj[0, 2] == 0.0
