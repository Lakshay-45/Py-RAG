FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on

# Create a non-root user and group (appuser will have UID/GID 1000)
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd --gid ${GROUP_ID} appuser && \
    useradd --uid ${USER_ID} --gid ${GROUP_ID} --shell /bin/bash --create-home appuser

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    python3-dev \
    gosu \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app
COPY ./entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create parent directories for volume mounts.
# These will be owned by root initially.
# The entrypoint script running as root will chown them.
RUN mkdir -p /app/chroma_db_data && \
    mkdir -p /app/persistent_metadata && \
    mkdir -p /app/data/uploads

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
# CMD for uvicorn will be passed to entrypoint.sh
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]