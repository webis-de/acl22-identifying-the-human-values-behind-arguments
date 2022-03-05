FROM anibali/pytorch:1.10.2-nocuda-ubuntu20.04

# Copy the requirement file to the container 
ADD /requirements.txt /app/
RUN pip install -r requirements.txt

# Copies your code file from your action repository to the filesystem path `/app/` of the container
COPY entrypoint.sh /app/entrypoint.sh

# Code file to execute when the docker container starts up (`entrypoint.sh`)
ENTRYPOINT ["/app/entrypoint.sh"]