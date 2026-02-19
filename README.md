# Grok-Like AI Agent in ~400 Lines of Python (2026 Edition)

Build your own witty, tool-using, Indic-language-capable AI agent inspired by Grok — using only Python, LangChain, and llama.cpp.

Features:
- ReAct-style agent with tools (web search + Python REPL)
- Runs efficiently on CPU (or GPU) via llama.cpp GGUF models
- Fine-tuning support for Indic languages (Hindi, Marathi, etc.)
- Deployable on AWS EC2 Mumbai region (low-latency for Indian users)
- Total core logic < 400 lines

Perfect for Mumbai/Bengaluru devs who want local-first AI without spending ₹50k/month on cloud GPUs.

## Quick Start (Local)

1. Install dependencies
```bash
pip install -r requirements.txt
```

Download Llama 3.1 8B GGUF (Q5_K_M recommended)
```bash
huggingface-cli download TheBloke/Llama-3.1-8B-GGUF llama-3.1-8b.Q5_K_M.gguf --local-dir ./models
```

Run simple chat test
```bash
python inference_test.py
```

Run full FastAPI server
```bash
uvicorn app:app --reload --port 8000
```

Then visit http://localhost:8000/docs and try the /chat endpoint.

Indic Fine-Tuning (Optional)
```bash
python fine_tune_indic.py
```

Requires GPU or patience (takes ~4–12 hours on t4g.medium EC2).
Deploy on AWS Mumbai (ap-south-1)
```bash
bash deploy_ec2.sh
```

See deploy_ec2.sh for full instructions.
