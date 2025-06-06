FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd --gid ${GROUP_ID} appuser && \
    useradd --uid ${USER_ID} --gid ${GROUP_ID} --shell /bin-false --create-home appuser

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app

# Copy the entrypoint script
COPY ./entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create data directories and set ownership during build
# This ensures they exist with correct ownership if not volume mounted initially
# or if volume mounts are to non-existent host paths (Docker creates them as root)
RUN mkdir -p /app/chroma_db_data && \
    mkdir -p /app/persistent_metadata && \
    mkdir -p /app/data/uploads

# Change ownership of the /app directory
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
# Use an entrypoint script
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
 # Default command for entrypoint