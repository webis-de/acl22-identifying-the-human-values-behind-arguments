# acl22-identifying-the-human-values-behind-arguments

## Content

### Argument Directory

The default directory is [`data`](data).
The argument files are the same as of [https://doi.org/10.5281/zenodo.5657250](https://doi.org/10.5281/zenodo.5657250).

* [`arguments.tsv`](data/arguments.tsv): Each row corresponds to one argument
  * `Argument ID`: The unique identifier for the argument
  * `Part`: Name of the containing dataset part from the paper; used in the [evaluation script](#evaluate-the-predictions)
  * `Usage`: Name of the set the argument is used for in the machine learning experiments; one of "train", "validation" or "test"; if this column is absent all arguments are counted for "test"
  * `Premise`: Premise text of the argument; the only input used for classification, `Conclusion` and `Stance` are currently ignored
* [`labels-level1.tsv`](data/labels-level1.tsv) / [`labels-level2.tsv`](data/labels-level2.tsv) / [`labels-level3.tsv`](data/labels-level3.tsv) / [`labels-level4a.tsv`](data/labels-level4a.tsv) / [`labels-level4b.tsv`](data/labels-level4b.tsv): Each row corresponds to one argument
  * `Argument ID`: The unique identifier for the argument
  * Other: The column name specifies a label in that level, and the value whether the argument has that label (1) or not (0)
* [`value.json`](data/values.json):
  * `levels`: The identifier for each level used in training and prediction
  * Other: The label names and order corresponding to each level

<img src="./markups/info-markup-requires-retraining.svg" alt="Changes of labels in the value.json require a re-training">

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
  * `-a, --argument-dir string` specifies the folder containing the argument files... (default is `./data/`)
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
  * `-a, --argument-dir string` specifies the folder containing the argument files... (default is `./data/`)
  * `-h, --help` displays the help text
  * `-m, --model-dir string` specifies the folder containing the trained models (default is `./data/models`)
  * `-o, --one-baseline` requests the prediction from 1-Baseline
  * `-s, --svm` requests the prediction from trained SVM

<img src="./markups/warning-markup-pickle.svg" alt="Only use serialize files from trustworthy sources.">

Example command:

```sh
sudo docker run --rm -it --init \
--volume="$PWD:/app" \
acl22_values:no_cuda python predict.py -a "./custom_dir/corpus" -s
```

### Evaluate the predictions

```bash
$ Rscript Evaluation.R [OPTIONS]
```

* `OPTIONS` are:
  * `-a, --argument-dir string` specifies the folder containing the prediction and argument files... (default is `./data/`)
  * `--absent-labels` to include the absent labels on each dataset part into the evaluation
  * `-h, --help` displays the help text
