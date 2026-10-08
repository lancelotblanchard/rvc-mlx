"""PyTorch reference implementations used only by the test-suite."""

import warnings

import torch


def legacy_weight_norm(module, name="weight", dim=0):
    """
    `torch.nn.utils.weight_norm` (deprecated hook API). Released RVC / fairseq checkpoints were saved with it, so their
    keys are `*.weight_g` / `*.weight_v`; the references use it on purpose to reproduce that layout.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return torch.nn.utils.weight_norm(module, name=name, dim=dim)
