# Use PyTorch CUDA runtime for GPU support
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Build args for user setup
ARG uid=1000
ARG gid=1000
ARG user=jnavarro

# Set up working directory inside the container, isolated from host mounts
WORKDIR /app

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

# Ensure user owns the isolated application directory
# Also create /araia so it exists for the volume mount
RUN chown -R ${user}:${user} /app && mkdir -p /araia && chown -R ${user}:${user} /araia

# Switch to non-root user
USER ${user}

# Copy Pixi manifest and lockfile
COPY --chown=${user}:${user} pyproject.toml pixi.lock ./
# Note: we MUST copy src/ here so the dynamic version resolution works during install
COPY --chown=${user}:${user} src/ ./src/

# Install environment in /app/.pixi 
# This is physically isolated from the /araia volume mount, so it won't be hidden
RUN pixi install 

# --- SURGICAL PYTORCH INJECTION ---
# Pixi installs its own CPU-only torch. We delete it and symlink the system (GPU) one.
# Base image stores torch in /opt/conda/lib/python3.11/site-packages/
RUN export SITE_PACKAGES=$(pixi run python -c "import site; print(site.getsitepackages()[0])") && \
    rm -rf ${SITE_PACKAGES}/torch* && \
    rm -rf ${SITE_PACKAGES}/nvidia* && \
    ln -s /opt/conda/lib/python3.11/site-packages/torch* ${SITE_PACKAGES}/ && \
    ln -s /opt/conda/lib/python3.11/site-packages/nvidia* ${SITE_PACKAGES}/
# ----------------------------------

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
             cd /araia && PYTHONPATH=/araia/src:/opt/conda/lib/python3.11/site-packages pixi run --manifest-path /app/pyproject.toml streamlit run src/modules/Welcome.py --server.port=8502 --server.address=0.0.0.0"

