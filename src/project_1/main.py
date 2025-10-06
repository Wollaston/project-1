"""
The main script for training the model and display the evaluation results.

Instructions:
---
Commonly, the main script contains following functions:
* ``train_loop``
* ``test_loop``
* Evaluation functions
* (Optional) Command line arguments or options
    * If you need explicit control over this script (e.g. learning rate, training size, etc.)
* (Optional) Any functions from ``utils.py`` that helps display results and evaluation

Eventually, this script should be run as
```
uv run main.py <ARGUMENTS> --<OPTIONS>
```

References:
---
https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
"""

import argparse
import logging
import uuid
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from project_1.dataset import DenseDataset, SparseDataset
from project_1.model import CNN, MLP, LogisticRegression
from project_1.utils import accuracy, precision, recall, sense_levels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Metrics(BaseModel):
    loss: float
    accuracy: float
    precision: float
    recall: float


class Experiment(BaseModel):
    metrics: Metrics
    model: Literal["logistic_regression", "mlp", "cnn"]
    sense: Literal[1, 2, 3]
    dimensions: Literal[50, 100, 200]
    encoding: Literal["sparse", "dense"]
    tensor: Literal["random", "glove"]
    learning_rate: float
    batch_size: int
    epochs: int

    def file_path(self) -> str:
        return f"{self.model}_{self.tensor}_{self.encoding}_{self.dimensions}_{self.sense}_{uuid.uuid4()}.json"


# Reference for train/test loops: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation
def train_loop(
    dataloader: DataLoader,
    model: LogisticRegression | MLP | CNN,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: Optimizer,
    batch_size: int,
):
    logging.info("Entering train loop")
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        logging.debug(f"Train Batch {batch}")
        # Compute prediction and loss
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            logging.info(f"loss: {loss:>7f}  [{current:>5d}]")


def test_loop(
    dataloader: DataLoader,
    model: LogisticRegression | MLP | CNN,
    loss_fn: nn.CrossEntropyLoss,
) -> Metrics:
    logging.info("Entering test loop")
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    test_loss, test_acc, test_prec, test_recall = 0, 0, 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred = model(X).argmax(1)
            test_acc += accuracy(pred, y)
            test_prec += precision(pred, y, 5)
            test_recall += recall(pred, y, 5)

    test_loss /= num_batches
    test_acc /= num_batches
    test_prec /= num_batches
    test_recall /= num_batches
    logging.info(
        f"Test Metrics: \n Avg Accuracy: {(test_acc * 100):>0.1f}%\n Avg Precision: {test_prec:>8f}\n Avg Recall: {test_recall:>8f}\n Avg Loss: {test_loss:>8f}"
    )
    return Metrics(
        loss=test_loss, accuracy=test_acc, precision=test_prec, recall=test_recall
    )


def cli() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("train_path", help="path to the train set")
    parser.add_argument("dev_path", help="path to the dev set")
    parser.add_argument("test_path", help="path to the test set")

    parser.add_argument(
        "-m",
        "--model",
        help="model to train",
        default="logistic_regression",
        choices=["logistic_regression", "mlp", "cnn"],
    )
    parser.add_argument(
        "-d",
        "--dimensions",
        help="embedding dimensions",
        default=50,
        choices=[50, 100, 200],
        type=int,
    )
    parser.add_argument(
        "-s",
        "--sense_level",
        help="pdtb sense level",
        default=1,
        type=int,
        choices=[1, 2, 3],
    )
    parser.add_argument(
        "-t",
        "--tensor_type",
        help="type of tensor to use",
        default="random",
        choices=["random", "glove"],
    )
    parser.add_argument(
        "-n",
        "--encoding",
        help="encoding type",
        choices=["sparse", "dense"],
    )
    parser.add_argument(
        "-p",
        "--glove_path",
        help="path to pre-trained GloVe embeddings",
    )

    parser.add_argument(
        "-l", "--learning_rate", help="learning rate", default="0.05", type=float
    )
    parser.add_argument(
        "-b", "--batch_size", help="batch size", default="128", type=int
    )
    parser.add_argument(
        "-e", "--epochs", help="number of epochs", default="32", type=int
    )
    parser.add_argument(
        "--log_level", help="log level", default="info", choices=["debug", "info"]
    )

    args = parser.parse_args()

    train_path = args.train_path
    dev_path = args.dev_path
    test_path = args.test_path

    model_type: Literal["logistic_regression", "mlp", "cnn"] = args.model
    dimensions: Literal[50, 100, 200] = args.dimensions
    level: Literal[1, 2, 3] = args.sense_level
    tensor: Literal["random", "glove"] = args.tensor_type
    encoding: Literal["sparse", "dense"] = args.encoding
    glove_path: Optional[str] = args.glove_path

    learning_rate: float = args.learning_rate
    batch_size: int = args.batch_size
    epochs: int = args.epochs
    log_level: Literal["debug", "info"] = args.log_level

    match log_level:
        case "debug":
            logging.basicConfig(
                level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
            )
        case "info":
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
            )

    logging.info(f"""Running {__name__} with:
Learning Rate: {learning_rate}
Batch Size: {batch_size}
Epochs: {epochs}""")

    if glove_path:
        glove = Path(glove_path)
    else:
        glove = None

    match encoding:
        case "dense":
            train_dataset = DenseDataset.from_path(
                corpus_path=Path(train_path),
                level=level,
                tensor=tensor,
                dimensions=dimensions,
                glove_path=glove,
            )
            logging.debug(
                f"Loaded Train Dataset. Vocab Size: {train_dataset.vocab_size}. Embeddings: {train_dataset.embeddings.weight.size()[0]}"
            )
            dev_dataset = DenseDataset.from_vocab(
                vocab=train_dataset.data,
                corpus_path=Path(dev_path),
                level=level,
                tensor=tensor,
                dimensions=dimensions,
                glove_path=glove,
            )
            dev_dataset.vocab_size = train_dataset.vocab_size
            dev_dataset.embeddings = train_dataset.embeddings
            logging.debug(
                f"Loaded Dev Dataset. Vocab Size: {dev_dataset.vocab_size}. Embeddings: {dev_dataset.embeddings.weight.size()[0]}"
            )

            test_dataset = DenseDataset.from_vocab(
                vocab=train_dataset.data,
                corpus_path=Path(test_path),
                level=level,
                tensor=tensor,
                dimensions=dimensions,
                glove_path=glove,
            )
            test_dataset.vocab_size = train_dataset.vocab_size
            test_dataset.embeddings = train_dataset.embeddings
            logging.debug(
                f"Loaded Test Dataset. Vocab Size: {test_dataset.vocab_size}. Embeddings: {test_dataset.embeddings.weight.size()[0]}"
            )

            match model_type:
                case "logistic_regression":
                    model = LogisticRegression(
                        input_size=dimensions * 3, output_size=sense_levels[level]
                    )
                case "mlp":
                    model = MLP(
                        input_size=dimensions * 3,
                        output_size=sense_levels[level],
                    )
                case "cnn":
                    model = CNN(
                        input_size=dimensions * 3,
                        output_size=sense_levels[level],
                    )
        case "sparse":
            train_dataset = SparseDataset.from_path(
                Path("./pdtb/train.json"), level=level
            )
            dev_dataset = SparseDataset.from_vocab(
                vocab=train_dataset.vocab, level=level
            )

            test_dataset = SparseDataset.from_vocab(
                vocab=train_dataset.vocab, level=level
            )

            match model_type:
                case "logistic_regression":
                    model = LogisticRegression(
                        input_size=train_dataset.vocab_size
                        + train_dataset.bigram_size
                        + 1,
                        output_size=sense_levels[level],
                    )
                case "mlp":
                    model = MLP(
                        input_size=train_dataset.vocab_size
                        + train_dataset.bigram_size
                        + 1,
                        output_size=sense_levels[level],
                    )
                case "cnn":
                    model = CNN(
                        input_size=train_dataset.vocab_size
                        + train_dataset.bigram_size
                        + 1,
                        output_size=sense_levels[level],
                    )

            logging.debug(f"Vocab Size: {train_dataset.vocab_size}")
            logging.debug(f"Train Dataset: {train_dataset}")

    train_dataloader: DataLoader[DenseDataset] = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    dev_dataloader: DataLoader[DenseDataset] = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_dataloader: DataLoader[DenseDataset] = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for t in range(epochs):
        logging.info(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        test_loop(dev_dataloader, model, loss_fn)
    logging.info("Done Training!")

    logging.info("Testing trained model against test set")
    metrics = test_loop(test_dataloader, model, loss_fn)
    experiment = Experiment(
        metrics=metrics,
        model=model_type,
        sense=level,
        dimensions=dimensions,
        tensor=tensor,
        encoding=encoding,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
    )
    path = experiment.file_path()
    with open(path, "w") as file:
        file.write(experiment.model_dump_json())


if __name__ == "__main__":
    print(
        "This project should be run as a cli.\nTry: `uv run main.py` to access the cli."
    )
