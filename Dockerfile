FROM python:3.11-slim-trixie

WORKDIR /app

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Actualizar tooling Python que trae vulnerabilidades conocidas
RUN python -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    "wheel>=0.46.2" \
    "jaraco.context>=6.1.0"

RUN pip install --no-cache-dir \
    torch==2.12.1 \
    torchvision==0.27.1 \
    --index-url https://download.pytorch.org/whl/cu132

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir \
    -r /app/requirements.txt

ENV HF_HOME=/root/.cache/huggingface

COPY src /app/src

ENTRYPOINT ["python", "/app/src/forensic_search.py"]
