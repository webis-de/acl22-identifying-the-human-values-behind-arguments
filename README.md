# acl22-identifying-the-human-values-behind-arguments

## Content

## Usage

### Windows

Run the following commmand in an elevated command prompt:
```cmd
execute_windows.bat
```

### Linux

Run:
```bash
$ ./execute_linux.sh
```

### Manual Execution

```bash
$ docker build -f Dockerfile -t acl22_values:no_cuda .
```

```sh
sudo docker run --rm -it --init \
--volume="$PWD:/app" \
acl22_values:no_cuda python main.py
```

* `--volume="$PWD:/app"`: Mounts the current working directory into the container.
  The default working directory inside the container is `/app`.
