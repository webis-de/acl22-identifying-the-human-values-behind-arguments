# acl22-identifying-the-human-values-behind-arguments

Machine Learning scripts for the identification of human values behind arguments.

[![license](https://img.shields.io/github/license/webis-de/acl22-identifying-the-human-values-behind-arguments)](https://github.com/webis-de/acl22-identifying-the-human-values-behind-arguments/blob/main/LICENSE)

As a first approach on the automatic identification of human values behind arguments the scripts can be used to train
different machine learning models based on a
Multi-Level Value-Taxonomy
and classify new arguments in regards of the human values they concern.

The available machine learning models are currently Bert and two baseline models (SVM and 1-Baseline).

## Data Directory

The default data directory is [`data`](data).
The example argument files are the same as of
[https://doi.org/10.5281/zenodo.5657250](https://doi.org/10.5281/zenodo.5657250).

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

In order to use the [image](#creating-the-docker-image) you must have Docker Engine installed. Instructions
for setting up Docker Engine are
[available on the Docker website](https://docs.docker.com/engine/installation/).

(Optional) In order to use the
[evaluation script](#evaluate-the-predictions)
on the made predictions you need an additional installation of
[R](https://cran.r-project.org/).

#### CUDA requirements

If you have a CUDA-compatible NVIDIA graphics card, you can use a CUDA-enabled
version of the PyTorch image to enable hardware acceleration.
<!-- This was only tested under ... -->

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
$ docker build -f Dockerfiles/Dockerfile_nocuda -t ghcr.io/webis-de/acl22-value-classification:nocuda .
```

If you want to use the cuda-enabled version instead run:

```bash
$ docker build -f Dockerfiles/Dockerfile_cuda11_3 -t ghcr.io/webis-de/acl22-value-classification:cuda11.3 .
```

### Train the models in the Docker container

In order to train the models you need the following files in your
[data directory](#data-directory):

* `arguments.tsv` with the required columns
  * `Argument ID`
  * `Premise`
  * `Usage` being "train" for each training argument (and "validation" for each validation argument)
* `value.json`
* `labels-level[X].tsv` for each level `X` used in `value.json`

After that you can run the training script:

```sh
docker run --rm -it --init \
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
* `IMAGE_NAME` is the name of your Docker image
* `OPTIONS` are:
  * `-c, --classifier string` specifies the classifiers to train: `b` for Bert, `s` for SVM (default is `b`)
  * `-d, --data-dir string` specifies the folder containing the argument files (default is `./data/`)
  * `-h, --help` displays the help text
  * `-m, --model-dir string` specifies the folder for storing the trained models (default is `./data/models`)
  * `-v, --validate` validate models after training

Example command:

```sh
docker run --rm -it --init \
--volume="$PWD:/app" \
ghcr.io/webis-de/acl22-value-classification:nocuda python training.py -d "./custom_dir/corpus" -c "bs" -v
```

### Run the prediction script inside the Docker container

In order to predict the values of arguments with the trained models you need the following files in your
[data directory](#data-directory):

* `arguments.tsv` with the required columns
  * `Argument ID`
  * `Premise`
  * `Usage` being "test" for each argument you want to classify (can be omitted, to include all arguments)
* `value.json` with the same contents used for training

After that you can run the prediction script:

```sh
docker run --rm -it --init \
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
* `IMAGE_NAME` is the name of your Docker image
* `OPTIONS` are:
  * `-c, --classifier string` specifies the used classifiers: `b` for Bert, `s` for SVM, `o` for 1-Baseline (default is `b`)
  * `-d, --data-dir string` specifies the folder containing the argument files (default is `./data/`)
  * `-h, --help` displays the help text
  * `-m, --model-dir string` specifies the folder containing the trained models (default is `./data/models`)

| :warning: | The Support Vector Machines are loaded and saved using &#8220;pickle&#8221; to de-/serialize the data object from/to:<br/><code>[MODEL_DIR]/svm/svm_train_level[X].sav</code><br/> The files are only checked to contain &#8220;sklearn Pipelines&#8221;.<br/><br/>Only use serialized files from trustworthy sources. |
| :---: | :--- |

Example command:

```sh
docker run --rm -it --init \
--volume="$PWD:/app" \
ghcr.io/webis-de/acl22-value-classification:nocuda python predict.py -d "./custom_dir/corpus" -c "bso"
```

### Evaluate the predictions

To evaluate the predictions made by the models your `arguments.tsv` and your `labels-level[X].tsv` with the
corresponding true labels
[prediction step](#run-the-prediction-script-inside-the-docker-container)
as well as the resulting `predictions.tsv` need to be in your
[data directory](#data-directory).

If the column `Part` is present in your `arguments.tsv` the evaluation will be made separately for each
instance of this column.

After that you can run the evaluation script calculating for all used models individually the label-wise
_Precision_, _Recall_, _F1-Score_, and _Accuracy_ as well as their mean for each level:

```bash
$ Rscript Evaluation.R [OPTIONS]
```

* `OPTIONS` are:
  * `-a, --absent-labels` to include the absent labels on each dataset part into the evaluation
  * `-d, --data-dir string` specifies the folder containing the prediction and argument files (default is `./data/`)
  * `-h, --help` displays the help text
