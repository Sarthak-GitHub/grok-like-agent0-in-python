#!/usr/bin/env bash

echo "AWS Mumbai (ap-south-1) deployment helper"

cat << 'EOF'
1. Launch t3.medium or g4dn.xlarge in ap-south-1 (Ubuntu 22.04)
2. Allow port 8000 in Security Group (your IP or 0.0.0.0/0 for testing)
3. SSH in and run:

sudo apt update && sudo apt install -y python3.12 python3-pip git build-essential cmake
git clone https://github.com/Sarthak-GitHub/grok-like-agent0-in-python.git
cd grok-like-agent-python
pip install -r requirements.txt

# Download model
huggingface-cli download TheBloke/Llama-3.1-8B-GGUF llama-3.1-8b.Q5_K_M.gguf --local-dir ./models

# Optional: fine-tune (needs GPU instance)
# python fine_tune_indic.py

# Run server
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

# Production: use systemd / PM2 + nginx + certbot
EOF
