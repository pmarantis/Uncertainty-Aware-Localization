"""
Backwards-compatible shim for the Nomadic CSI dataset and datamodule.

The NomadicLocalizationDataset and NomadicLocalizationDataModule implementations
now live in the trustworthy-uq package (``trustworthy_uq.data.nomadic``). This
module re-exports them so existing code such as
``from dataset import NomadicLocalizationDataModule`` keeps working. Install the
package with ``pip install -r requirements-portable.txt``.
"""

from trustworthy_uq.data.nomadic import (  # noqa: F401
    NomadicLocalizationDataModule,
    NomadicLocalizationDataset,
)

__all__ = ["NomadicLocalizationDataset", "NomadicLocalizationDataModule"]
