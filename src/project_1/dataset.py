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

from itertools import pairwise
import logging
from pathlib import Path
from typing import Literal, Optional

import torch
from torch.nn import Embedding
from torch.utils.data import Dataset
from project_1.sample import Sample
from project_1.vocab import DenseVocab, SparseVocab
from .utils import map_senses


class DenseTransform(object):
    def __call__(
        self,
        sample: Sample,
        level: int,
        embeddings: Embedding,
        vocab: dict[str, int],
    ) -> tuple[torch.Tensor, int]:
        logging.debug(f"Transforming Sample {sample.doc_id}")

        arg1_embeddings_ids: list[int] = []
        for tok in set(sample.arg1.raw_text.split()):
            try:
                arg1_embeddings_ids.append(vocab[tok.lower()])
            except Exception as e:
                logging.debug(f"Token {tok} not in vocab. Using '<unk>' token\n{e}")
                arg1_embeddings_ids.append(vocab["<unk>"])
        arg1_embeddings_lookup = torch.LongTensor(sorted(arg1_embeddings_ids))
        logging.debug(f"Arg1 Embeddings Lookup: {arg1_embeddings_lookup}")

        arg2_embeddings_ids: list[int] = []
        for tok in set(sample.arg2.raw_text.split()):
            try:
                arg2_embeddings_ids.append(vocab[tok.lower()])
            except Exception as e:
                logging.debug(f"Token {tok} not in vocab. Using '<unk>' token\n{e}")
                arg2_embeddings_ids.append(vocab["<unk>"])
        arg2_embeddings_lookup = torch.LongTensor(sorted(arg2_embeddings_ids))
        logging.debug(f"Arg2 Embeddings Lookup: {arg2_embeddings_lookup}")

        connective_embeddings_ids: list[int] = []
        if sample.connective.raw_text == "":
            connective_embeddings_ids.append(vocab["<unk>"])
        else:
            for tok in set(sample.connective.raw_text.split()):
                try:
                    connective_embeddings_ids.append(vocab[tok.lower()])
                except Exception as e:
                    logging.debug(f"Token {tok} not in vocab. Using '<unk>' token\n{e}")
                    connective_embeddings_ids.append(vocab["<unk>"])
        connective_embeddings_lookup = torch.LongTensor(
            sorted(connective_embeddings_ids)
        )
        logging.debug(f"Connective Embeddings Lookup: {connective_embeddings_lookup}")

        arg1_embedding: torch.Tensor = self.average_embeddings(
            embeddings(arg1_embeddings_lookup)
        )
        arg2_embedding: torch.Tensor = self.average_embeddings(
            embeddings(arg2_embeddings_lookup)
        )
        connective_embedding: torch.Tensor = self.average_embeddings(
            embeddings(connective_embeddings_lookup)
        )

        tensor = torch.cat((arg1_embedding, arg2_embedding, connective_embedding), 0)

        logging.debug(f"Transformed Tensor to Length {len(tensor)}")
        return (tensor, map_senses(sense=sample.sense[0], level=level))

    def average_embeddings(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        logging.debug(f"Averaging {len(tensors)} Tensors")
        return torch.mean(torch.stack(tuple(tensors), dim=0), dim=0)


class DenseDataset(Dataset):
    """Dataset class for the PDTB dataset"""

    def __init__(
        self,
        corpus_path: Path,
        vocab: DenseVocab,
        level: Literal[1, 2, 3],
        tensor: Literal["random", "glove"],
        dimensions: Literal[50, 100, 200],
        glove_path: Optional[Path] = None,
    ):
        logging.info(f"""Loading Dataset with:
Path: {corpus_path}
Level: {level}
Tensor: {tensor}
dimensions: {dimensions}
glove_path: {glove_path}
""")
        self.data: DenseVocab = vocab
        self.vocab: dict[str, int] = vocab.vocab
        self.transform = DenseTransform()
        self.level = level
        self.tensor = tensor

        match tensor:
            case "random":
                self.vocab["<unk>"] = len(self.vocab) - 1
                self.vocab_size = len(self.vocab)
                self.embeddings = Embedding(self.vocab_size, dimensions)
                logging.debug(f"Vocab Size: {self.vocab_size}")
                logging.debug(f"Embeddings: {self.embeddings}")
            case "glove":
                if not glove_path:
                    raise ValueError(
                        "'glove_path' must be provided to generate glove embeddings from file"
                    )
                try:
                    embeddings: dict[int, torch.Tensor] = {}
                    self.vocab["<unk>"] = len(self.vocab)
                    updated_vocab: dict[str, int] = {}
                    idx = 0
                    with open(glove_path, "r") as file:
                        for line in file:
                            glove_data = line.strip().split()
                            tok = glove_data[0]
                            if (
                                tok.lower() not in updated_vocab
                                and tok.lower() in self.vocab
                            ):
                                updated_vocab[tok.lower()] = idx
                                idx += 1
                            try:
                                tok_idx = self.vocab[tok]
                                if tok_idx not in embeddings:
                                    embedding = torch.FloatTensor(
                                        [
                                            float(val)
                                            for idx, val in enumerate(glove_data[1:])
                                            if idx < dimensions
                                        ]
                                    )
                                    embeddings[tok_idx] = embedding
                            except Exception:
                                pass
                    self.embeddings = Embedding.from_pretrained(
                        torch.stack(tuple(embeddings.values()))
                    )
                    self.vocab = updated_vocab
                    self.vocab_size = self.embeddings.weight.size()[0]
                    logging.debug(f"Embeddings: {self.embeddings}")
                    logging.debug(f"Vocab: {len(self.vocab)}")
                    logging.debug(f"Vocab Size: {self.vocab_size}")
                except Exception as e:
                    raise ValueError(
                        f"Could not open GloVe embeddings at path {glove_path}\n{e}"
                    )

    @classmethod
    def from_vocab(
        cls,
        corpus_path: Path,
        vocab: DenseVocab,
        level: Literal[1, 2, 3],
        tensor: Literal["random", "glove"],
        dimensions: Literal[50, 100, 200],
        glove_path: Optional[Path] = None,
    ) -> "DenseDataset":
        """Class constructor to simplify loading a dataset from a path.
        Assumes file is encoded as JSON Lines."""
        return cls(
            vocab=vocab,
            level=level,
            corpus_path=corpus_path,
            tensor=tensor,
            dimensions=dimensions,
            glove_path=glove_path,
        )

    @classmethod
    def from_path(
        cls,
        corpus_path: Path,
        level: Literal[1, 2, 3],
        tensor: Literal["random", "glove"],
        dimensions: Literal[50, 100, 200],
        glove_path: Optional[Path] = None,
    ) -> "DenseDataset":
        """Class constructor to simplify loading a dataset from a path.
        Assumes file is encoded as JSON Lines."""
        vocab = DenseVocab.from_path(corpus_path)

        return cls(
            vocab=vocab,
            level=level,
            corpus_path=corpus_path,
            tensor=tensor,
            dimensions=dimensions,
            glove_path=glove_path,
        )

    def __len__(self):
        return len(self.data.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        logging.debug(f"Instance IDX: {idx}")
        return self.transform(
            sample=self.data.data[idx],
            level=self.level,
            embeddings=self.embeddings,
            vocab=self.vocab,
        )


class SparseTransform(object):
    def __call__(
        self, vocab: SparseVocab, sample: Sample, vocab_size: int, level: int
    ) -> tuple[torch.Tensor, int]:
        X = torch.zeros(vocab_size + 1)
        for tok in sample.vocab:
            if tok in vocab.vocab:
                X[vocab.vocab[tok]] += 1
            else:
                X[-1] += 1

        for bigram in pairwise(sample.vocab):
            if bigram in vocab.bigrams:
                X[vocab.bigrams[bigram]] += 1

        y = map_senses(sense=sample.sense[0], level=level)

        return (X, y)


class SparseDataset(Dataset):
    """Dataset class for the PDTB dataset"""

    def __init__(self, vocab: SparseVocab, level: int):
        self.vocab = vocab
        self.vocab_size: int = len(self.vocab.vocab)
        self.bigram_size: int = len(self.vocab.bigrams)
        self.transform = SparseTransform()
        self.level = level

    def __len__(self):
        return len(self.vocab.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        return self.transform(
            self.vocab,
            self.vocab.data[idx],
            self.vocab_size + self.bigram_size,
            level=self.level,
        )

    @classmethod
    def from_vocab(cls, vocab: SparseVocab, level: int) -> "SparseDataset":
        """Class constructor to simplify loading a dataset from a path.
        Assumes file is encoded as JSON Lines."""
        return cls(vocab=vocab, level=level)

    @classmethod
    def from_path(cls, corpus_path: Path, level: int) -> "SparseDataset":
        """Class constructor to simplify loading a dataset from a path.
        Assumes file is encoded as JSON Lines."""
        vocab = SparseVocab.from_path(corpus_path)
        return cls(vocab=vocab, level=level)
