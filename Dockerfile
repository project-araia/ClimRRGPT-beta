# Use Ubuntu base image
FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Build args for user setup
ARG uid=1000
ARG gid=1000
ARG user

# Environment variables (must come after ARG)
ENV uid=${uid}
ENV gid=${gid}
ENV user=${user}

# Set up working directory
WORKDIR /callm

# Add deadsnakes PPA for Python 3.11
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    vim \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Create user
RUN groupadd -g ${gid} ${user}
RUN useradd -m -u ${uid} -g ${gid} -s /bin/bash ${user}

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | bash

# Expose Ollama and Streamlit ports
EXPOSE 11434 8501

# Switch to non-root user
USER ${user}

# Ensure ~/.local/bin is in PATH
ENV PATH="~/.local/bin:$PATH"

# Copy requirements and install Python deps
COPY --chown=${user}:${user} requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

CMD bash -c "ollama serve > /tmp/ollama.log 2>&1 & \
	     sleep 2; \
             if ! ollama list | grep -q 'qwen3'; then \
                echo 'Model not found, pulling...'; \
                ollama pull qwen3 2>&1 | tee /dev/stdout; \
             else \
                echo 'Model already present, skipping pull.'; \
             fi; \
             echo 'Starting server at: http://localhost:8501'; \
             streamlit run src/modules/Welcome.py --server.port=8501 --server.address=0.0.0.0"
