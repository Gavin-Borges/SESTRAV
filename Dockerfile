# Production-grade Dockerfile optimized for SESTRAV Geometric Deep Learning (GNN)
# Pins stable, non-breaking releases of PyTorch, CUDA 12.1, and PyTorch Geometric.

# Using official PyTorch runtime with CUDA 12.1 pre-installed to ensure maximum performance and zero GPU driver drift.
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

LABEL org.opencontainers.image.authors="SESTRAV Project Contributors"
LABEL description="SESTRAV-CORE — HPC-Optimized Geometric Graph Neural Network Pipeline"

# Set non-interactive timezone/locale installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for compilation and standard scientific tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch Geometric and its compiled dependencies matching PyTorch 2.2.1 and CUDA 12.1
# Using the pre-compiled wheel indexes from pyg.org ensures no slow C++ compiles on host and exact version alignment.
RUN pip install --no-cache-dir \
    torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.2.1+cu121.html \
    torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.2.1+cu121.html \
    torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.2.1+cu121.html \
    torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-2.2.1+cu121.html \
    torch-geometric==2.5.2

# Copy python requirements file and install additional dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline and library source code
COPY . .

# Run dynamic verification checks to ensure key libraries import correctly
RUN python -c "import torch; import torch_geometric; print('CUDA Available:', torch.cuda.is_available()); print('PyG Version:', torch_geometric.__version__)"

# Expose non-privileged execution user for security compliance on multi-tenant HPC systems
RUN useradd -m -s /bin/bash sestrav && chown -R sestrav:sestrav /app
USER sestrav

# Expose default healthcheck verifying PyG and torch are functional
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import torch_geometric" || exit 1

ENTRYPOINT ["python"]
CMD ["src/verify/sestrav_evaluator.py", "src/verify/targets.json", "--mock"]
