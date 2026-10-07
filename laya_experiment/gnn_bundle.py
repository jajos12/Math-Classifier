"""Compatibility container for the published checkpoint implementation.

The initial implementation keeps model-specific code separate from the public
pipeline API. The graph, data, and Laya modules can be tested without loading
Torch or downloading the checkpoint.
"""

from .graph import *  # noqa: F401,F403


def load_bundle(*args, **kwargs):
    raise NotImplementedError(
        "Published GNN architecture loading is the next implementation step; "
        "graph conversion and experiment interfaces are now available."
    )


def predict_probs(*args, **kwargs):
    raise NotImplementedError("GNN inference is not available until the checkpoint backend is installed.")


def topk_predictions(probs, bundle, k):
    import torch
    k = min(k, probs.shape[-1])
    values, indices = torch.topk(probs, k, dim=-1)
    names = [[bundle.id_to_tactic[int(index)] for index in row] for row in indices]
    return names, [[float(value) for value in row] for row in values]
