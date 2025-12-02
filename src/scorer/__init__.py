import json
from .linear import LinearScorer
from .tree   import TreeScorer
from .ensemble import WeightedEnsemble

def build_scorer(cfg):
    # 1️⃣ Linear
    with open(cfg['scorer']['linear_weights_path']) as f:
        lin_weights = json.load(f)
    lin = LinearScorer(lin_weights)

    # 2️⃣ Tree
    with open(cfg['scorer']['tree_feature_order']) as f:
        feat_order = json.load(f)
    tree = TreeScorer(cfg['scorer']['tree_model_path'], feat_order)

    # 3️⃣ Ensemble
    comps = [
        (lin,  cfg['scorer']['ensemble_weights']['linear']),
        (tree, cfg['scorer']['ensemble_weights']['tree']),
    ]
    return WeightedEnsemble(comps)
