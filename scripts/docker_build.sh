#!/usr/bin/env bash
set -euo pipefail

# Config (edit here)
# Image name and tag
IMAGE_NAME="${IMAGE_NAME:-torch-dev}"
IMAGE_TAG="${IMAGE_TAG:-cu128}"
# CUDA and Ubuntu Versions
CUDA_VERSION="${CUDA_VERSION:-12.8.1}"
UBUNTU_VERSION="${UBUNTU_VERSION:-22.04}"
# Python version
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
# PyTorch versions
TORCH_CUDA="${TORCH_CUDA:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"

# Match host UID/GID to avoid permission issues on bind mounts
UID_VAL="${UID_VAL:-$(id -u)}"
GID_VAL="${GID_VAL:-$(id -g)}"

# Optional: disable cache with NO_CACHE=1
NO_CACHE="${NO_CACHE:-0}"
CACHE_FLAG=""
if [[ "${NO_CACHE}" == "1" ]]; then
  CACHE_FLAG="--no-cache"
fi

# Build
docker build ${CACHE_FLAG} \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg UBUNTU_VERSION="${UBUNTU_VERSION}" \
  --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
  --build-arg TORCH_CUDA="${TORCH_CUDA}" \
  --build-arg TORCH_VERSION="${TORCH_VERSION}" \
  --build-arg TORCHVISION_VERSION="${TORCHVISION_VERSION}" \
  --build-arg UID="${UID_VAL}" \
  --build-arg GID="${GID_VAL}" \
  -f docker/Dockerfile \
  .

echo "Done: ${IMAGE_NAME}:${IMAGE_TAG}"
