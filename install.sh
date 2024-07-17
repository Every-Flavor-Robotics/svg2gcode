#!/bin/bash

# Ensure the script runs from the correct directory
# Ensure the script runs from the correct directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build the two packages
/root/.cargo/bin/cargo build --release -p svg2gcode-cli -p optimizer



