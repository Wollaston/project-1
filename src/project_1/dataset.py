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

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset
from project_1.sample import Sample
from project_1.vocab import PDTBVocab
from utils import map_senses


class TransformSample(object):
    def __call__(self, sample: Sample, level: int) -> tuple[torch.Tensor, int]:
        if sample.tensor:
            return (sample.tensor, map_senses(sense=sample.sense[0], level=level))
        else:
            raise ValueError(f"Cannot transform sample {sample}. No tensor.")


class PDTBDataset(Dataset):
    """Dataset class for the PDTB dataset"""

    def __init__(
        self, corpus_path: Path, level: int, tensor: Literal["random", "glove"]
    ):
        self.vocab: PDTBVocab = PDTBVocab.from_path(corpus_path)
        self.transform = TransformSample()
        self.level = level
        self.tensor = tensor

    def __len__(self):
        return len(self.vocab.data)

    def __getitem__(self, idx):
        return self.transform(self.vocab.data[idx], self.level)

    def tokenize(self, instance: str) -> list[str]:
        return instance.lower().strip().split()
