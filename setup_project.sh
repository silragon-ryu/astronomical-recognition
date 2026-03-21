#!/bin/bash

PROJECT_NAME="planet-bias-analysis"

echo "Creating project structure..."

mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

# Root folders
mkdir -p data/{raw,processed,disturbed}
mkdir -p datasets
mkdir -p training
mkdir -p experiments/configs
mkdir -p results/{plots,logs,checkpoints,images}
mkdir -p notebooks

# Models
mkdir -p models/mlp
mkdir -p models/cnn_reg
mkdir -p models/transfer_learning

# Dashboard
mkdir -p dashboard/app/routes
mkdir -p dashboard/app/templates
mkdir -p dashboard/static/js

# ----------------------------
# Create core files
# ----------------------------

touch README.md
touch requirements.txt
touch .gitignore

# Dataset files
touch datasets/loader.py
touch datasets/transforms.py
touch datasets/utils.py

# Training files
touch training/trainer.py
touch training/metrics.py
touch training/logger.py

# Experiment runner
touch experiments/run_all.py

# Models
touch models/mlp/model.py
touch models/mlp/train.py
touch models/mlp/config.yaml

touch models/cnn_reg/model.py
touch models/cnn_reg/train.py
touch models/cnn_reg/config.yaml

touch models/transfer_learning/model.py
touch models/transfer_learning/train.py
touch models/transfer_learning/config.yaml

# Dashboard backend
touch dashboard/app/main.py
touch dashboard/app/routes/metrics.py
touch dashboard/app/routes/predictions.py
touch dashboard/app/routes/images.py

# Dashboard frontend
touch dashboard/app/templates/index.html
touch dashboard/static/js/app.js

# Notebook
touch notebooks/analysis.ipynb

# Results files
touch results/metrics.json
touch results/predictions.csv

# ----------------------------
# .gitignore
# ----------------------------
cat <<EOL > .gitignore
__pycache__/
*.pyc
.env
data/
results/checkpoints/
node_modules/
EOL

# ----------------------------
# README
# ----------------------------
cat <<EOL > README.md
# Planet Bias Analysis

## Goal
Investigate whether neural networks truly learn planetary features or rely on dataset bias.

## Models
- MLP (baseline)
- CNN + Regularization
- Transfer Learning

## Experiments
- Raw dataset
- Debiased dataset
- Disturbed dataset

## Dashboard
FastAPI + Tailwind visualization
EOL

# ----------------------------
# requirements.txt
# ----------------------------
cat <<EOL > requirements.txt
torch
torchvision
fastapi
uvicorn
jinja2
pandas
matplotlib
opencv-python
EOL

echo "Project structure created successfully."