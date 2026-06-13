#!/bin/bash
set -e

PROD_DB="/penny/data/penny/penny.db"

# Create production backup only when SNAPSHOT=1 (set by make up / make prod)
if [ "${SNAPSHOT:-0}" = "1" ] && [ -f "$PROD_DB" ]; then
    BACKUP_DIR="/penny/data/penny/backups"
    mkdir -p "$BACKUP_DIR"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="$BACKUP_DIR/penny.db.$TIMESTAMP"
    echo "Creating database snapshot: $BACKUP_FILE"
    cp "$PROD_DB" "$BACKUP_FILE"

    # Keep only last 5 backups
    ls -t "$BACKUP_DIR"/penny.db.* 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null
fi

# Execute the main command
exec "$@"
