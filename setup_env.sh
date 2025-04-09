#!/bin/bash

# -- Directories
SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# -- Files
INI_FILE="$SCRIPT_DIR/settings.ini"
REQ_FILE="$SCRIPT_DIR/requirements.txt"
VENV_DIR="$PROJECT_DIR/venv"

# -- Variables
CARLA_PATH=""
found_section=false

# -- Parse settings.ini
while IFS='=' read -r line value; do
    line=$(echo "$line" | tr -d ' ') # Trim leading spaces
    
    if [[ "$line" == "[CARLA-SERVER]" ]]; then
        found_section=true
    elif [[ "$line" == "[CARLA-CLIENT]" ]]; then
        found_section=false
    elif [[ "$found_section" == "true" ]]; then
        if [[ "$line" == "CARLA_PATH_WHL" ]]; then
            CARLA_PATH=$(echo "$value" | tr -d ' ') # Trim leading spaces
        fi
    fi
done < "$INI_FILE"

echo ""

if [[ -z "$CARLA_PATH" ]]; then
    echo "ERROR: CARLA_PATH not found in settings.ini."
    exit 1
fi

echo "CARLA_PATH: $CARLA_PATH"
echo "Creating virtual environment in: $VENV_DIR"

python3.7 -m venv "$VENV_DIR"
if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to create virtual environment. Make sure Python 3.7 is installed."
    exit 1
fi

source "$VENV_DIR/bin/activate"
if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to activate the virtual environment."
    exit 1
fi

# -- Install the CARLA wheel file
echo "Installing CARLA wheel from: $CARLA_PATH"
pip install "$CARLA_PATH"
if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to install CARLA egg: $CARLA_PATH"
    exit 1
fi

pip install -r "$REQ_FILE"

echo ""
echo "Done. Changing directory to: $PROJECT_DIR"
cd "$PROJECT_DIR"