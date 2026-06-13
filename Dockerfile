FROM python:3.13-slim@sha256:f82c96458eedc847b233e582eb31336f4954b39cae020b6dcf5b3ed0e5cbcd74

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create user and working directory
RUN useradd -m -s /bin/bash sestrav_user
WORKDIR /app
RUN chown -R sestrav_user:sestrav_user /app

# Switch to non-root user
USER sestrav_user

# Set up python environment variables
ENV PATH="/home/sestrav_user/.local/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

# Copy source and config
COPY --chown=sestrav_user:sestrav_user pyproject.toml README.md ./
COPY --chown=sestrav_user:sestrav_user src/ ./src/
COPY --chown=sestrav_user:sestrav_user config.yaml ./

# Install SESTRAV package
RUN pip install --user "pip==26.1.2" && \
    pip install --user .

# Default command
ENTRYPOINT ["sestrav"]
CMD ["--help"]
