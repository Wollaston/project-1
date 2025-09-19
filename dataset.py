"""
The dataset module contains any dataset representation as subclass of
``torch.utils.data.Dataset``.

Instruction:
---
To give you full flexibility to implement your preprocessing pipeline,
the only class provided is ``PDTBDataset`` which is required to put in
``torch.utils.data.DataLoader``.

Other than this class, you're free and welcome to implement any function
or class needed.
"""

from torch.utils.data import Dataset


class PDTBDataset(Dataset):
    """Dataset class for the PDTB dataset"""
    raise NotImplementedError