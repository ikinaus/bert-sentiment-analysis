FROM nvcr.io/nvidia/tritonserver:24.10-py3

COPY requirements.txt /tmp/requirements.txt

RUN pip3 install --no-cache-dir -r /tmp/requirements.txt