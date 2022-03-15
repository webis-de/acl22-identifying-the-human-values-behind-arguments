# acl22-identifying-the-human-values-behind-arguments

Introduction

## Content

### Argument Directory

The default argument directory is [`data`](data).
The argument files are the same as of [https://doi.org/10.5281/zenodo.5657250](https://doi.org/10.5281/zenodo.5657250).

* [`arguments.tsv`](data/arguments.tsv): Each row corresponds to one argument
  * `Argument ID`: The unique identifier for the argument
  * `Part`: Name of the containing dataset part from the paper; used in the [evaluation script](#evaluate-the-predictions)
  * `Usage`: Name of the set the argument is used for in the machine learning experiments; one of "train", "validation" or "test"; if this column is absent all arguments are counted for "test"
  * `Premise`: Premise text of the argument; the only input used for classification, `Conclusion` and `Stance` are currently ignored
* [`value.json`](data/values.json):
  * `levels`: The identifier for each level used in training and prediction
  * Other: The label names and order corresponding to each level
* [`labels-level1.tsv`](data/labels-level1.tsv) / [`labels-level2.tsv`](data/labels-level2.tsv) / [`labels-level3.tsv`](data/labels-level3.tsv) / [`labels-level4a.tsv`](data/labels-level4a.tsv) / [`labels-level4b.tsv`](data/labels-level4b.tsv): Each row corresponds to one argument
  * `Argument ID`: The unique identifier for the argument
  * Other: The column name specifies a label in that level (must be the same as in `value.json`), and the value whether the argument has that label (1) or not (0)

| :information_source: | If you change the label order, add, remove, or edit any of the labels in `value.json` you would need to re-train the models in order to make predictions. |
| :---: | :--- |

## Usage

### Requirements

In order to use this image you must have Docker Engine installed. Instructions
for setting up Docker Engine are
[available on the Docker website](https://docs.docker.com/engine/installation/).

(Optional) In order to use the
[evaluation script](#evaluate-the-predictions)
on the made predictions you need an additional installation of
[R](https://cran.r-project.org/).

#### CUDA requirements

If you have a CUDA-compatible NVIDIA graphics card, you can use a CUDA-enabled
version of the PyTorch image to enable hardware acceleration. This was only
tested under T.B.A.

Firstly, ensure that you install the appropriate NVIDIA drivers. You should use
an installed version of CUDA _at least as new as the image you intend to use_
(currently `CUDA 11.3`) via
[the official NVIDIA CUDA download page](https://developer.nvidia.com/cuda-downloads).

You will also need to install the NVIDIA Container Toolkit to enable GPU device
access within Docker containers. This can be found at
[NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

### Creating the Docker image

In the main directory create the image by running:

```bash
$ docker build -f Dockerfiles/Dockerfile_nocuda -t acl22_values:no_cuda .
```

If you want to use the cuda-enabled version instead run:

```bash
$ docker build -f Dockerfiles/Dockerfile_cuda11_3 -t acl22_values:cuda11.3 .
```

### Train the models in the Docker container

In order to train the models you need the following files in your
[argument directory](#argument-directory):

* `arguments.tsv` with the required columns
  * `Argument ID`
  * `Premise`
  * `Usage` being "train" for each training argument (and "validation" for each validation argument)
* `value.json`
* `labels-level[X].tsv` for each level `X` used in `value.json`

After that you can run the training script:

```sh
sudo docker run --rm -it --init \
--gpus=all \
--volume="$PWD:/app" \
IMAGE_NAME python training.py [OPTIONS]
```

* `--gpus=all`: Required if using CUDA, optional otherwise. Passes the
  graphics cards from the host to the container. You can also more precisely
  control which graphics cards are exposed using this option (see documentation
  at https://github.com/NVIDIA/nvidia-docker).
* `--volume="$PWD:/app"`: Mounts the current working directory into the container.
  The default working directory inside the container is `/app`.
* `IMAGE_NAME` is the name of your Docker image (either `acl22_values:no_cuda` or `acl22_values:cuda11.3`)
* `OPTIONS` are:
  * `-a, --argument-dir string` specifies the folder containing the argument files (default is `./data/`)
  * `-h, --help` displays the help text
  * `-m, --model-dir string` specifies the folder containing the trained models (default is `./data/models`)
  * `-s, --svm` requests the prediction from trained SVM
  * `-v, --validate` validate models after training

Example command:

```sh
sudo docker run --rm -it --init \
--volume="$PWD:/app" \
acl22_values:no_cuda python training.py -a "./custom_dir/corpus" -s -v
```

### Run the prediction script inside the Docker container

In order to predict the values of arguments with the trained models you need the following files in your
[argument directory](#argument-directory):

* `arguments.tsv` with the required columns
  * `Argument ID`
  * `Premise`
  * `Usage` being "test" for each argument you want to classify (can be omitted, to include all arguments)
* `value.json` with the same contents used for training

After that you can run the prediction script:

```sh
sudo docker run --rm -it --init \
--gpus=all \
--volume="$PWD:/app" \
IMAGE_NAME python predict.py [OPTIONS]
```

* `--gpus=all`: Required if using CUDA, optional otherwise. Passes the
  graphics cards from the host to the container. You can also more precisely
  control which graphics cards are exposed using this option (see documentation
  at https://github.com/NVIDIA/nvidia-docker).
* `--volume="$PWD:/app"`: Mounts the current working directory into the container.
  The default working directory inside the container is `/app`.
* `IMAGE_NAME` is the name of your Docker image (either `acl22_values:no_cuda` or `acl22_values:cuda11.3`)
* `OPTIONS` are:
  * `-a, --argument-dir string` specifies the folder containing the argument files (default is `./data/`)
  * `-h, --help` displays the help text
  * `-m, --model-dir string` specifies the folder containing the trained models (default is `./data/models`)
  * `-o, --one-baseline` requests the prediction from 1-Baseline
  * `-s, --svm` requests the prediction from trained SVM

| :warning: | The Support Vector Machines are loaded and saved using &#8220;pickle&#8221; to de-/serialize the data object from/to:<br/><code>[MODEL_DIR]/svm/svm_train_level[X].sav</code><br/> The files are only checked to contain &#8220;sklearn Pipelines&#8221;.<br/><br/>Only use serialized files from trustworthy sources. |
| :---: | :--- |

Example command:

```sh
sudo docker run --rm -it --init \
--volume="$PWD:/app" \
acl22_values:no_cuda python predict.py -a "./custom_dir/corpus" -s
```

### Evaluate the predictions

To evaluate the predictions made by the models your `arguments.tsv` and your `labels-level[X].tsv` with the
corresponding true labels
[prediction step](#run-the-prediction-script-inside-the-docker-container)
as well as the resulting `predictions.tsv` need to be in your
[argument directory](#argument-directory).

After that you can run the evaluation script calculating for all used models individually the label-wise
`Precision`, `Recall`, `F1-Score`, and `Accuracy` as well as their mean for each level:

```bash
$ Rscript Evaluation.R [OPTIONS]
```

* `OPTIONS` are:
  * `-a, --argument-dir string` specifies the folder containing the prediction and argument files (default is `./data/`)
  * `--absent-labels` to include the absent labels on each dataset part into the evaluation
  * `-h, --help` displays the help text
