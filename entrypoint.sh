#!/bin/sh

set -e # Exit immediately if a command exits with a non-zero status.

# Define user and group IDs
APP_UID=1000
APP_GID=1000
APP_USER=appuser

# Define paths for persistent data
CHROMA_DATA_DIR="/app/chroma_db_data"
METADATA_DIR="/app/persistent_metadata"
UPLOADS_DIR="/app/data/uploads"

# Ensure directories exist
mkdir -p "$CHROMA_DATA_DIR"
mkdir -p "$METADATA_DIR"
mkdir -p "$UPLOADS_DIR"

echo "Entrypoint: Current user is $(whoami)"
echo "Entrypoint: Ensuring ownership of data directories for $APP_USER ($APP_UID:$APP_GID)..."

# Change ownership of the data directories to appuser.
chown -R "${APP_UID}:${APP_GID}" "$CHROMA_DATA_DIR"
chown -R "${APP_UID}:${APP_GID}" "$METADATA_DIR"
chown -R "${APP_UID}:${APP_GID}" "$UPLOADS_DIR"

echo "Entrypoint: Ownership set. Directories content:"
ls -ld "$CHROMA_DATA_DIR"
ls -ld "$METADATA_DIR"
ls -ld "$UPLOADS_DIR"

echo "Entrypoint: Dropping privileges and executing command as $APP_USER: $@"
exec gosu "$APP_USER" "$@"