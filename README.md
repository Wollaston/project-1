# Project 1

This project uses `uv` and is run as a cli.

Paths to the dataset and pre-trained embeddings are provided as parameters
to the cli. There is no required canonical or hard-coded path.

To run:

```shell
$ uv run main.py

usage: main.py [-h] [-m {logistic_regression,mlp,cnn}] [-d {50,100,200}] [-s {1,2,3}] [-t {random,glove}] [-n {sparse,dense}] [-p GLOVE_PATH]
               [-l LEARNING_RATE] [-b BATCH_SIZE] [-e EPOCHS] [--log_level {debug,info}]
               train_path dev_path test_path

positional arguments:
  train_path            path to the train set
  dev_path              path to the dev set
  test_path             path to the test set

options:
  -h, --help            show this help message and exit
  -m {logistic_regression,mlp,cnn}, --model {logistic_regression,mlp,cnn}
                        model to train
  -d {50,100,200}, --dimensions {50,100,200}
                        embedding dimensions
  -s {1,2,3}, --sense_level {1,2,3}
                        pdtb sense level
  -t {random,glove}, --tensor_type {random,glove}
                        type of tensor to use
  -n {sparse,dense}, --encoding {sparse,dense}
                        encoding type
  -p GLOVE_PATH, --glove_path GLOVE_PATH
                        path to pre-trained GloVe embeddings
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size
  -e EPOCHS, --epochs EPOCHS
                        number of epochs
  --log_level {debug,info}
                        log level
```
