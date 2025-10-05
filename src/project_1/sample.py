from pydantic import AliasPath, BaseModel, ConfigDict, Field, computed_field


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

    @computed_field()
    @property
    def vocab(self) -> set[str]:
        vocab: set[str] = set()
        [vocab.add(tok.lower()) for tok in self.arg1.raw_text.split()]
        [vocab.add(tok.lower()) for tok in self.arg2.raw_text.split()]
        [vocab.add(tok.lower()) for tok in self.connective.raw_text.split()]
        return vocab
