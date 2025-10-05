from itertools import pairwise
from pathlib import Path
from pydantic import BaseModel

from .sample import Sample


class DenseVocab(BaseModel):
    """Generate Vocabulary from PDTB dataset"""

    data: list[Sample]
    vocab: dict[str, int]

    @classmethod
    def from_path(
        cls,
        path: Path,
    ) -> "DenseVocab":
        """Class constructor to simplify loading a dataset from a path.
        Assumes file is encoded as JSON Lines."""
        vocab: dict[str, int] = {}

        idx = 0
        samples: list[Sample] = []
        with open(path, "r", encoding="utf8") as file:
            for line in file:
                sample = Sample.model_validate_json(line)
                samples.append(sample)

        for sample in samples:
            for tok in sample.vocab:
                if tok.lower() not in vocab:
                    vocab[tok.lower()] = idx
                    idx += 1

        return cls(data=samples, vocab=vocab)

    @classmethod
    def data_only(
        cls,
        path: Path,
        primary_vocab: dict[str, int],
    ):
        """Class constructor to simplify loading a dataset from a path.
        Assumes file is encoded as JSON Lines."""

        ## Get samples
        samples: list[Sample] = []
        with open(path, "r", encoding="utf8") as file:
            for line in file:
                sample = Sample.model_validate_json(line)
                samples.append(sample)

        return cls(vocab=primary_vocab, data=samples)


class SparseVocab(BaseModel):
    """Generate Vocabulary from PDTB dataset"""

    data: list[Sample]
    vocab: dict[str, int]
    bigrams: dict[tuple[str, str], int]

    @classmethod
    def from_path(cls, path: Path) -> "SparseVocab":
        """Class constructor to simplify loading a dataset from a path.
        Assumes file is encoded as JSON Lines."""
        vocab: dict[str, int] = {}
        bigrams: dict[tuple[str, str], int] = {}

        idx = 0
        ## Get samples
        samples: list[Sample] = []
        with open(path, "r", encoding="utf8") as file:
            for line in file:
                sample = Sample.model_validate_json(line)
                samples.append(sample)

        for sample in samples:
            for tok in sample.vocab:
                if tok not in vocab:
                    vocab[tok] = idx
                    idx += 1

        for sample in samples:
            for bigram in pairwise(sample.vocab):
                if bigram not in bigrams:
                    bigrams[bigram] = idx
                    idx += 1

        return cls(vocab=vocab, data=samples, bigrams=bigrams)

    @classmethod
    def data_only(
        cls,
        path: Path,
        primary_vocab: dict[str, int],
        primary_bigrams: dict[tuple[str, str], int],
    ):
        """Class constructor to simplify loading a dataset from a path.
        Assumes file is encoded as JSON Lines."""

        ## Get samples
        samples: list[Sample] = []
        with open(path, "r", encoding="utf8") as file:
            for line in file:
                sample = Sample.model_validate_json(line)
                samples.append(sample)

        return cls(vocab=primary_vocab, data=samples, bigrams=primary_bigrams)
