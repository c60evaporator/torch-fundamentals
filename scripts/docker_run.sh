#!/usr/bin/env bash
set -euo pipefail

# Config (edit here)
# Image name and tag
IMAGE_NAME="${IMAGE_NAME:-torch-dev}"
IMAGE_TAG="${IMAGE_TAG:-cu128}"
# Container name
CONTAINER_NAME="${CONTAINER_NAME:-pytorch_dev}"
# Project dir -> /workspace
WORKDIR_HOST="${WORKDIR_HOST:-$(pwd)}"
# Read-only mount for safety: RO=1 (default) / RO=0 for writable
RO="${RO:-1}"
DS_MOUNT_OPT="ro"
if [[ "${RO}" == "0" ]]; then
  DS_MOUNT_OPT="rw"
fi
# Jupyter port mapping
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
# GPU settings
GPUS="${GPUS:-all}"
# Shared memory (DataLoader multi-workers)
SHM_SIZE="${SHM_SIZE:-16g}"

# Sanity checks
if [[ ! -d "${WORKDIR_HOST}" ]]; then
  echo "ERROR: WORKDIR_HOST does not exist: ${WORKDIR_HOST}" >&2
  exit 1
fi

echo "Running container: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  WORKDIR_HOST=${WORKDIR_HOST} -> /workspace"
echo "  JUPYTER_PORT=${JUPYTER_PORT}"
echo "  GPUS=${GPUS}"

# Run the container
docker run --rm --init -it \
  --name "${CONTAINER_NAME}" \
  --gpus "${GPUS}" \
  --ipc=host \
  --shm-size="${SHM_SIZE}" \
  -p "${JUPYTER_PORT}:8888" \
  -v "${WORKDIR_HOST}:/workspace" \
  -e DATASET_ROOT="/workspace/data" \
  "${IMAGE_NAME}:${IMAGE_TAG}"
