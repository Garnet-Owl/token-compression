# Token Compression for LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An LLM-optimized text compression system that reduces token usage by **30-50%** while maintaining semantic understanding. The fine-tuned model can process compressed prompts directly, saving API costs.

> [!NOTE]
> This project is an experimental idea inspired by a video about ZIP compression. It explores the intersection of classical compression algorithms and LLM context window optimization.

This project is **open source** and contributions are welcome! feel free to open issues or submit pull requests.

## 🎯 Key Results

| Metric                | Value       |
| --------------------- | ----------- |
| Token Reduction       | 30-50%      |
| Compression Ratio     | 1.2x - 2.5x |
| Semantic Preservation | 100%        |

## 🏗️ Architecture

```
Input Text → Dictionary Compression → LZ77 Back-References → Compressed Tokens
                                                                    ↓
                                                          Fine-tuned LLM
                                                      (understands compressed format)
                                                                    ↓
                                                              Response
```

**Compression transforms:**

- `"the quick brown fox"` → `"①quick brown fox"` (dictionary)
- `"database connection pool ... database connection"` → `"database connection pool ... ‹45:19›"` (LZ77)

## 📦 Installation

- [uv](https://github.com/astral-sh/uv) package manager

### Environment Setup

1. **Install uv** (if not already installed)

   ```bash
   pip install uv
   ```

2. **Create a virtual environment with Python 3.12**

   ```bash
   uv venv --python=python3.12
   ```

3. **Activate the virtual environment**

   Windows:

   ```bash
   .venv\Scripts\activate
   ```

   Linux/macOS:

   ```bash
   source .venv/bin/activate
   ```

   ```

   ```

4. **Install dependencies with uv**
   ```bash
   uv sync
   ```
5. **Install the pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 🚀 Quick Start

### Start the API Server

```bash
python -m app.main_runner.api
```

### Make a Request

```bash
# PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/ask" -Method POST -ContentType "application/json" -Body '{"context": "Your long context here...", "question": "Your question?"}'

# Or open http://localhost:8000/docs for interactive Swagger UI
```

### Python Usage

```python
from app.main_runner.qa_service import QAService

service = QAService()
result = service.ask(
    context="The database connection pool manages all connections...",
    question="What does the pool manage?"
)
print(result["answer"])
print(f"Saved {result['token_reduction_pct']}% tokens")
```

## 📁 Project Structure

```
token-compression/
├── app/
│   ├── token_compression/     # Core compression (LZ77 + dictionary)
│   │   ├── compressor.py
│   │   ├── decompressor.py
│   │   └── service.py
│   ├── fine_tuning/           # LoRA training pipeline
│   │   ├── trainer.py
│   │   └── dataset.py
│   └── main_runner/           # API & inference
│       ├── api.py             # FastAPI server
│       ├── qa_service.py      # Q&A with compression
│       └── config.py
├── models/
│   └── lora_adapter/          # Fine-tuned LoRA weights
├── notebooks/
│   └── kaggle_finetune.ipynb  # Training notebook (Kaggle GPU)
└── data/
    └── training_pairs.json    # Training data
```

## 📐 Mathematical Specification

For the formal mathematical foundation including:

- LZ77 matching functions
- Compression-aware attention mechanism
- Lazy decompression theorem
- Token reduction analysis

See: **[Mathematical Spec.md](Mathematical%20Spec.md)**

## 🔧 How It Works

1. **Dictionary Compression**: Common phrases (`"the "`, `"and "`, etc.) → single tokens (`①`, `②`)
2. **LZ77 Back-References**: Repeated text → `‹distance:length›` format
3. **Fine-tuned LLM**: Qwen2.5-0.5B trained to understand compressed format
4. **LoRA Adapters**: Efficient fine-tuning with 0.4% trainable parameters

## 🏋️ Training

Training was done on Colab with GPU. See `notebooks/token_compression_colab_finetune.ipynb`.

```python
# Training config
LoRA: r=16, alpha=32, target=["q_proj", "k_proj", "v_proj", "o_proj"]
Epochs: 50
Batch size: 8
Data: 100 (original, compressed) pairs
```

## 📊 API Endpoints

| Endpoint  | Method | Description                                                   |
| --------- | ------ | ------------------------------------------------------------- |
| `/ask`    | POST   | Send `{context, question}`, get `{answer, compression_ratio}` |
| `/health` | GET    | Check model status                                            |
| `/docs`   | GET    | Interactive Swagger UI                                        |

## 👤 Author

**James Wanjiku**
AI Researcher @ [AAI-Labs](https://www.aai-labs.com/en/cases)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/james-wanjiku/)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
