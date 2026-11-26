"""
Backwards-compatible shim for the ADN backbone models.

The DenseNet and AttentionDenseNet implementations now live in the
trustworthy-uq package (``trustworthy_uq.localization.backbone``). This module
re-exports them so existing code such as ``from models import AttentionDenseNet``
keeps working. Install the package with ``pip install -r requirements-portable.txt``.
"""

from trustworthy_uq.localization.backbone import (  # noqa: F401
    AttentionDenseNet,
    DenseBlock,
    DenseNet,
    LinearBlock,
    SelfAttention,
)

__all__ = ["DenseBlock", "LinearBlock", "DenseNet", "SelfAttention", "AttentionDenseNet"]
