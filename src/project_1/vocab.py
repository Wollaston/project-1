from pathlib import Path
from pydantic import BaseModel

from project_1.sample import Sample


class PDTBVocab(BaseModel):
    """Generate Vocabulary from PDTB dataset"""

    data: list[Sample]

    @classmethod
    def from_path(cls, path: Path) -> "PDTBVocab":
        """Class constructor to simplify loading a dataset from a path.
        Assumes file is encoded as JSON Lines."""

        # TODO: Init tensors here, avoid None in Sample for it
        samples: list[Sample] = []
        with open(path, "r", encoding="utf8") as file:
            for line in file:
                sample = Sample.model_validate_json(line)
                samples.append(sample)

        return cls(data=samples)
