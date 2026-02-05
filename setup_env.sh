#!/bin/bash

# Load pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Navigate to project
cd '/mnt/c/Users/rsamarth/OneDrive - ZeOmega/Desktop/project_demo'

# Set Python 3.12 and create venv
pyenv local 3.12
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete!"
python --version
