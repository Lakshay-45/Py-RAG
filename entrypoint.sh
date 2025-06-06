#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Ensure the directory for the SQLite database exists and is writable.
DB_DIR="/app/persistent_metadata"
DB_FILE="${DB_DIR}/metadata.db"

# Create the directory if it doesn't exist.
echo "Entrypoint: Checking if $DB_DIR is writable by $(whoami)..."
if [ ! -w "$DB_DIR" ]; then
    echo "Error: $DB_DIR is not writable. Attempting to chown..."
fi

echo "Entrypoint: Starting application with command: $@"

# Execute the command passed as arguments to the entrypoint (e.g., uvicorn)
exec "$@"