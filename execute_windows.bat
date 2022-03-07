docker build -f Dockerfile -t acl22_values:no_cuda ./

docker run --rm -it --init --volume="%CD%:/app" acl22_values:no_cuda python main.py