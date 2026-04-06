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

# Install essential system dependencies and certificates
RUN apt-get update && apt-get install -y \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Pixi binary
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

# Create non-root user
RUN groupadd -g ${gid} ${user} \
    && useradd -m -u ${uid} -g ${gid} -s /bin/bash ${user}

# Install Ollama (binary available for Linux x86_64)
RUN curl -fsSL https://ollama.com/install.sh | bash

# Ensure user owns the application directory and home
RUN chown -R ${user}:${user} /araia

# Switch to non-root user for Pixi install and runtime
USER ${user}
ENV PATH="/home/${user}/.pixi/bin:$PATH"

# Copy Pixi manifest and lockfile
COPY --chown=${user}:${user} pyproject.toml pixi.lock ./

# Install environment from lockfile
# Note: Pixi handles Linux-64 specific binaries automatically
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
