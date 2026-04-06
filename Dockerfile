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

# Move Pixi environment out of the mounted volume to avoid host permission conflicts
# /home/jnavarro/.pixi will be inside the container's writable filesystem
ENV PIXI_HOME=/home/${user}/.pixi
ENV CONDA_PREFIX=/home/${user}/.pixi/envs/default
ENV PIXI_PROJECT_MANIFEST=/araia/pyproject.toml
ENV PATH="/home/${user}/.pixi/envs/default/bin:/home/${user}/.pixi/bin:/usr/local/bin:$PATH"

# Copy Pixi manifest and lockfile
COPY --chown=${user}:${user} pyproject.toml pixi.lock ./
COPY --chown=${user}:${user} src/ ./src/

# Install environment (this will now live in /home/jnavarro/.pixi/envs)
# We use --manifest-path to be explicit
RUN pixi install -a --frozen

# Copy rest of the source code
COPY --chown=${user}:${user} . .

# Expose Streamlit (using 8502) and Ollama ports
EXPOSE 8502 11434

# Use 'pixi run' to ensure the environment is correctly loaded
# Note: Added 15s sleep for Ollama and fixed port to 8502
CMD bash -c "export PATH=$PATH:/usr/local/bin && \
             ollama serve > /tmp/ollama.log 2>&1 & \
             sleep 15; \
             if ! ollama list > /dev/null 2>&1; then \
                echo 'Ollama still starting, waiting...'; \
                sleep 15; \
             fi; \
             if ! ollama list | grep -q 'qwen3'; then \
                echo 'Model not found, pulling qwen3...'; \
                ollama pull qwen3 || true; \
             fi; \
             echo 'Starting ClimRRGPT-beta via Pixi on port 8502...'; \
             pixi run --manifest-path /araia/pyproject.toml streamlit run src/modules/Welcome.py --server.port=8502 --server.address=0.0.0.0"
