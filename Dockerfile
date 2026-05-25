FROM python:3.11-slim-bookworm@sha256:8dca233de9f3d9bb410665f00a4da6dd06f331083137e0e98ccf227236fcc438

LABEL org.opencontainers.image.authors="SESTRAV Project Contributors"
LABEL description="SESTRAV — Structural Epitope Scoring via TCR Recognition And Vaccinology"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY environments/requirements.lock ./environments/
RUN pip install --no-cache-dir -r environments/requirements.lock

COPY . .

RUN python -c "from src.features import TRAIN_FEATURE_COLUMNS, FEATURE_COLUMNS_30; assert len(TRAIN_FEATURE_COLUMNS) == 21; assert len(FEATURE_COLUMNS_30) == 30"

RUN useradd -m -s /bin/bash sestrav && chown -R sestrav:sestrav /app
USER sestrav

RUN mhcflurry-downloads fetch models_class1_presentation

ENTRYPOINT ["python"]
CMD ["pipeline.py"]
