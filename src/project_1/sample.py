from typing import Optional

from pydantic import AliasPath, BaseModel, ConfigDict, Field
from torch import Tensor


class Arg(BaseModel):
    """Pydantic BaseModel that represents an `arg
    field in a `Sample`."""

    raw_text: str = Field(validation_alias=AliasPath("RawText"))


class Sample(BaseModel):
    """Pydantic BaseModel that represents a sample
    in a dataset.

    Pydantic is used as it has built-in efficient and
    correct JSON utilities."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    arg1: Arg = Field(validation_alias=AliasPath("Arg1"))
    arg2: Arg = Field(validation_alias=AliasPath("Arg2"))
    connective: Arg = Field(validation_alias=AliasPath("Connective"))
    doc_id: str = Field(validation_alias=AliasPath("DocID"))
    id: int = Field(validation_alias=AliasPath("ID"))
    sense: list[str] = Field(validation_alias=AliasPath("Sense"))
    tensor: Optional[Tensor] = None
