# Use PyTorch CUDA runtime for GPU support
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Build args for user setup
ARG uid=1000
ARG gid=1000
ARG user=jnavarro

# Set up working directory
WORKDIR /araia

# Install manual system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Pixi binary globally to /usr/local/bin
RUN curl -fsSL https://pixi.sh/install.sh | bash \
    && mv /root/.pixi/bin/pixi /usr/local/bin/pixi

# Create non-root user
RUN groupadd -g ${gid} ${user} \
    && useradd -m -u ${uid} -g ${gid} -s /bin/bash ${user}

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | bash

# Set up Pixi directory with correct permissions for the user
RUN mkdir -p /araia && chown -R ${user}:${user} /araia

# Switch to non-root user
USER ${user}
WORKDIR /araia

# Copy only requirements first for caching
COPY --chown=${user}:${user} pyproject.toml pixi.lock ./

# Install environment
RUN pixi install

# Copy source code
COPY --chown=${user}:${user} . .

# Expose Streamlit and Ollama ports
EXPOSE 8501 11434

# Use 'pixi run' to ensure the environment is correctly loaded
CMD bash -c "ollama serve > /tmp/ollama.log 2>&1 & \
             sleep 5; \
             if ! ollama list | grep -q 'qwen3'; then \
                echo 'Model not found, pulling qwen3...'; \
                ollama pull qwen3 2>&1 || true; \
             else \
                echo 'Model already present, skipping pull.'; \
             fi; \
             echo 'Starting ClimRRGPT-beta via Pixi...'; \
             pixi run streamlit run src/modules/Welcome.py --server.port=8501 --server.address=0.0.0.0"
