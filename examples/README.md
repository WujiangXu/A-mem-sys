# Example Using A-mem-sys

I used:
- Self-hosted ollama server (hosted by my client)
- Self-hosted ChromaDB server (in Docker)

```
cp .env.example .env

# Edited the .env file to add my values

python -m venv .venv
pip install -r requirements.txt
python -m examples.main
```