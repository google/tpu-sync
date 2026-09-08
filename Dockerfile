# Use the official ML Build container as the base
FROM us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest

# Switch to root to install system packages
USER root

# Install clang and llvm which are required by XLA/tpu-raiden's bazel configuration
RUN apt-get update && apt-get install -y clang llvm && rm -rf /var/lib/apt/lists/*

# The container will run as the default user inherited from ml-build
