#!/bin/bash
# Documentation development server script

set -e

echo "Starting MkDocs development server..."
echo "Documentation will be available at: http://127.0.0.1:8000"
echo ""
echo "The server will automatically reload when files change in:"
echo "  - doc/docs/ (documentation files)"
echo "  - src/ (Python source code with docstrings)"
echo "  - config/ (configuration files)"
echo "  - mkdocs.yml (MkDocs configuration)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Change to project root directory
cd "$(dirname "$0")/.."

# Start MkDocs development server
uv run mkdocs serve --dev-addr=127.0.0.1:8000