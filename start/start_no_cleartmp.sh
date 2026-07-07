#!/bin/bash
set -euo pipefail

export CLEANUP_INTERVAL_SECONDS=315360000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../start.sh" "$@"
