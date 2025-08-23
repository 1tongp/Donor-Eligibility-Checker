# Donor Health Summary & RAG Chat (Local, Gradio + OpenAI)

A local, minimal project to learn core AI components: **RAG** + **Health Summary generation** with **citations**.  
UI is **Gradio**; models use **OpenAI** (LLM + embeddings). All data is **synthetic**.

## Quickstart
```bash
python3 -m venv venv
source venv/bin/activate
```
1. Create a Python 3.11+ virtual environment and install deps:
   ```bash
   pip install -r requirements.txt
   ```

2. Export your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```
   (Windows PowerShell: `$env:OPENAI_API_KEY="sk-..."`)

3. Build the RAG index (for policy docs):
   ```bash
   python app/build_index.py
   ```

4. Launch the Gradio app:
   ```bash
   python app/app_gradio.py
   ```

## What’s inside
- **data/donors.csv**: Synthetic donor health data (no real PHI).
- **data/policy_docs/**: RAG knowledge base — simplified eligibility rules & FAQ.
- **index/faiss/**: Vector index persisted by LlamaIndex (after step 3).
- **app/**: All Python code (ingest/index/summarise/chat/Gradio UI).

## Notes
- This is **not** medical software. It provides general information only and avoids diagnosis.
- All data is synthetic; thresholds in policy docs are simplified for learning purposes.


## Extras
- **app/synthesize_data.py**: regenerate `data/donors.csv` with synthetic values & injected edge cases.
  - Example: `python app/synthesize_data.py --n 300 --seed 123`
- **config/guardrails.yaml** + **app/guardrails.py**: simple red-flag detection (e.g., "chest pain").
  - You can edit the YAML to customize phrases and messages.


## Setting up local RAG + LLM
```bash
brew services start ollama
brew services list   # 可看到 ollama 状态应为 started
```

```bash 
# 建议轻量模型，适合 CPU
ollama pull qwen2.5:3b
```

```bash
# 快速自测
ollama run qwen2.5:3b
# 出现交互后，输入一行测试问题，Ctrl+C 退出
```

Then we can start the app for local RAG and LLM settings. Always remember to re-build the index after switching the local/online mode 