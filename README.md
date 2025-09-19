# Project 1 (Due: 10/3/2025)
This is the first project in this semester, and you're going to train several neural models with different architectures 
on Penn Discourse Tree Bank (PDTB). The goal of this project is to practice your deep learning programming skills and 
learn from observations produced by practical training.

## Starter code structure
The project skeleton has the following structure:

```
├── dataset.py
├── main.py
├── model.py
├── pyproject.toml
├── README.md
├── utils.py
└── uv.lock
```

## Setup

### Prerequisites
You must make sure of following software are installed:
* `uv`
* `git`

### Clone the starter code
Run the following commands to clone starter code and prepare for your own branch to implement the project:

```bash
git clone <repo>                      # Clone starter code to local

cd <project_dir>                      # Change to project directory path
git checkout -b <your_last_name>_pa0  # Create and switch to your feature branch
```

### Download dataset
In Moodle, download the `.zip` file named `project_0_data` to your local machine and unzip it to the project
directory.

> [!IMPORTANT]
> Make sure to not push dataset to GitHub!


### Python environment
You're given very needed external libraries as follows:
* `numpy`
* `torch ~= 2.6.0`
* `tqdm`

Normally you don't need other external libraries/packages, so make sure you really need them before add.

When the first time you entered the project directory, sync and install dependencies:
```bash
uv sync
```

To add a library/package:
```bash
uv add <pkg_name>
```
To remove a library/package:
```bash
uv remove <pkg_name>
```

## Instructions
The project has three components to complete:
1. Implementations of data preprocessing and model building (same as project 0)
2. Experiments that guide you to learn the difference between various settings
3. A writeup for your findings, which follows an academic writing convention

### Dataset
You'll use the exactly same Penn Discourse Tree Bank (PDTB) dataset as project 0.

### Implementations
Overall, you'll need to implement but not limited to below files:
* `dataset.py`: the module for implementing and represent data corpus in a subclass of `torch.utils.data.Dataset`
* `model.py`: the module for implementing neural models which are subclass of `torch.nn.Module`
* `utils.py` (Optional): the module containing utility functions/classes 
* `main.py`: the script to train and evaluate the model as well as running experiments

To run the whole thing, simply execute:
```bash
uv run main.py <ARGUMENTS> --<OPTIONS>
```

### Models
You have the following models to implement in the module `model.py`:
* A softmax classifier same as you did in project 0
* A multilayer perceptron (MLP) with your decided number of hidden layers and size
* A convolutional neural network with your customized configuration (e.g. paddings, strides)

Eventually, these models will be imported in `main.py` for training, evaluation and experiments.

### Word embeddings
Different from project 0 where you encoded tokens into sparse vectors, you're required to 
use dense vectors instead in this project. Mathematically, a dense vector is high-dimensional vector with most non-zeros, 
and is typically much smaller than a sparse vector in length. The mechanism of encoding a token into a dense 
vector is applying an embedding function to its token ID. Remember that a token is sparsely encoded by referring to its index 
from the vocabulary, then build a one-hot vector for it. Similarly, encoding the same token into a dense vector follows 
the same mechanism but in different output.

In this project, you'll need to implement two embeddings in `dataset.py`:
* Random initialized embeddings: use `torch.nn.Embedding` to map token IDs to dense vectors
* Pretrained embeddings (GloVe): download GloVe embeddings and process it to create corresponding mapping that is used for encoding

_*Source*_:
* To use random embeddings, read the doc about [`torch.nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
* Download GloVe from [here](https://nlp.stanford.edu/projects/glove/)

### Training and evaluation
Implement a training loop and a test loop as we discussed in the recitation, and you can find the notebook 
[here](https://github.com/Brandeis-CS231/recitation/blob/main/recitation_2_build_train_model.ipynb). You'll still use 
accuracy, precision and recall as your evaluation metrics. Additionally, you need to implement hyperparameter tuning 
(e.g. grid search) to find the optimal model setting. 
You can implement this part in `main.py`, or in a new module if you prefer.

### Experiments
Since you've built fundamental pieces in previous sections, now it's time to run several experiments to both test your 
implementations and find the difference across model architectures and feature representations.

#### Experiment 1: Effect of (random) dense representations
You'll start the first experiment by exploring how the dense vectors may affect training the MLP. Now that embeddings 
can be divided into sparse and dense, it's interesting to compare how model performs on these two different representations.

*Your tasks*:
- Train both sparse and dense representations on two MLPs with exactly same hyperparameters
- Experiment different number of hidden sizes AND hidden layers to optimize evaluation metrics

#### Experiment 2: Effect of pre-trained embeddings
Except for investigating (randomly) initialized embeddings, you are asked to investigate how the pre-trained embeddings 
influence the model's performance compared to the random embeddings. 
You'll use GloVe embeddings with hidden sizes $50, 100, 200$. Therefore, you need to use same hidden sizes for the random 
embeddings.

*Your tasks*:
- Train two MLPs using both pre-trained and random embeddings with three different hidden sizes
- Compare how the model performs for two different embeddings with the same hidden size
- Compare how the model performs for the same embeddings with the different hidden sizes

#### Experiment 3: Model architectures
Given you have three models (logistic regression, MLP and CNN), train each of them using GloVe embeddings with the size 
you found optimal in experiment 2 and maximize values of evaluation metrics as much as possible. 
Then, pick the best setting for each model and observe their corresponding evaluation results.

### Write-up
The writeup needs to follow an academic writing convention including below components:
* Introduction
* Methods
* Results
* Analysis
* Conclusion

Eventually, the write-up should not exceed 5 pages!

## Submission
When you're ready to submit, push your branch to GitHub and submit a pull request (PR). The submission
includes two things:
1. Implemented codes
2. A short finding as a `.pdf` file

> [!IMPORTANT]
> You'll not receive any extension on this project, your quota of late days will be accordingly reduced if you submit late.

## Rubrics
The grading is based on below parts:
* Complete and successful implementations (50)
  * Data preprocessing (20)
  * Model build (10)
  * Training & evaluation (20)
* Experiments (30)
* Writeup (10)
* Programming habits (10)
