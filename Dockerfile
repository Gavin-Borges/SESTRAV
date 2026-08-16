FROM python:3.13-slim@sha256:f82c96458eedc847b233e582eb31336f4954b39cae020b6dcf5b3ed0e5cbcd74

LABEL org.opencontainers.image.title="SESTRAV" \
      org.opencontainers.image.description="Structural Epitope Scoring via TCR Recognition And Vaccinology - MHC class I viral immunogenicity scoring CLI. Does not include the six-stage Snakemake workflow, which runs from a source checkout." \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/Gavin-Borges/SESTRAV" \
      org.opencontainers.image.documentation="https://github.com/Gavin-Borges/SESTRAV/blob/main/USAGE.md"

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

# Copy source and config.
#
# All THREE packaged trees must be copied. pyproject.toml declares
# packages.find include = ["sestrav*", "src*", "functions*"], but setuptools can
# only find what is in the build context: until 2026-08-14 this stage copied
# src/ alone, so the image silently installed a third of the package. `sestrav
# --help` still worked (the console script is src.cli:main), which is why the
# docker.yml smoke test would not have caught it, but `import sestrav` failed
# outright and `sestrav predict` died at stage 1, since src/cli.py cmd_predict
# imports all four functions/stage*.py modules.
COPY --chown=sestrav_user:sestrav_user pyproject.toml README.md ./
COPY --chown=sestrav_user:sestrav_user src/ ./src/
COPY --chown=sestrav_user:sestrav_user functions/ ./functions/
COPY --chown=sestrav_user:sestrav_user sestrav/ ./sestrav/
# config.yaml is copied for anyone running from /app, but note that the
# INSTALLED package will not read it: src/cli.py:_read_config resolves
# config.yaml relative to the package directory (site-packages), and the file
# ships in neither the wheel nor an sdist because there is no MANIFEST.in and it
# sits outside every declared package. _read_config swallows that as an empty
# dict, so `sestrav info` reports no config rather than erroring. Tracked
# separately - packaging config.yaml as package data is a wider change than this
# image.
COPY --chown=sestrav_user:sestrav_user config.yaml ./

# Install SESTRAV package
RUN pip install --user "pip==26.1.2" && \
    pip install --user .

# Default command
ENTRYPOINT ["sestrav"]
CMD ["--help"]
