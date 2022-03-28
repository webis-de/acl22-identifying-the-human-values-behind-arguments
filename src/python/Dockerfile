ARG CUDA=nocuda
FROM anibali/pytorch:1.10.2-${CUDA}-ubuntu20.04

COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY components/ /app/components
COPY predict.py training.py /app/
RUN python predict.py --help
