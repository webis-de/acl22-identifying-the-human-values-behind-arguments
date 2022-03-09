# acl22-identifying-the-human-values-behind-arguments

## Content

## Usage

### Creating the Docker image

```bash
$ docker build -f Dockerfile_nocuda -t acl22_values:no_cuda .
```

```bash
$ docker build -f Dockerfile_cuda -t acl22_values:cuda .
```

### Train the models in the Docker container

```sh
sudo docker run --rm -it --init \
--volume="$PWD:/app" \
IMAGE_NAME python training.py [OPTIONS]
```

* `--volume="$PWD:/app"`: Mounts the current working directory into the container.
  The default working directory inside the container is `/app`.
* `IMAGE_NAME` is the name of your Docker image (either `acl22_values:no_cuda` or `acl22_values:cuda`)
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
--volume="$PWD:/app" \
IMAGE_NAME python predict.py [OPTIONS]
```

* `--volume="$PWD:/app"`: Mounts the current working directory into the container.
  The default working directory inside the container is `/app`.
* `IMAGE_NAME` is the name of your Docker image (either `acl22_values:no_cuda` or `acl22_values:cuda`)
* `OPTIONS` are:
  * `-a, --argument-dir string` specifies the folder containing the argument files... (default is `./data/`)
  * `-h, --help` displays the help text
  * `-m, --model-dir string` specifies the folder containing the trained models (default is `./data/models`)
  * `-o, --one-baseline` requests the prediction from 1-Baseline
  * `-s, --svm` requests the prediction from trained SVM

Example command:

```sh
sudo docker run --rm -it --init \
--volume="$PWD:/app" \
acl22_values:no_cuda python predict.py -a "./custom_dir/corpus" -s
```