FROM nvcr.io/nvidia/pytorch:25.05-py3

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG MINIFORGE_VERSION=24.11.3-0
ENV CONDA_DIR=/opt/miniforge
ENV PATH=/opt/miniforge/envs/etsam/bin:${CONDA_DIR}/bin:${PATH}

WORKDIR /workspace/etsam

# Install Miniforge (includes conda and mamba).
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget aria2 bzip2 ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    wget -qO /tmp/miniforge.sh "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" && \
    bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" && \
    rm -f /tmp/miniforge.sh && \
    conda config --system --set auto_update_conda false

# Copy only the environment first to leverage Docker layer caching.
# The project source is bind-mounted by the devcontainer, not copied.
COPY environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml && \
    conda clean -afy

# The devcontainer runs as the base image's existing non-root `ubuntu` user
# (UID/GID 1000) and remaps it to the host user (remoteUser + updateRemoteUserUID),
# so generated files are owned by the host. No extra user is created here to avoid
# colliding with that UID.
