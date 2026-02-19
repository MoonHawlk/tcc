# Comparison of LLM Models for Data Retrieval from Static Pages Using RAG Architecture

## Overview

This work presents a comparative analysis of data retrieval from static pages using Large Language Models (LLMs) and a Retrieval-Augmented Generation (RAG) architecture. The study evaluates multiple Ollama-served models against Wikipedia pages in different languages, using the RAGAS framework for automated scoring.

The pipeline loads a web page, splits it into chunks, indexes them via embeddings in a Chroma vector store, and uses a RetrievalQA chain to answer batches of questions from CSV files. Results are evaluated with RAGAS metrics (faithfulness and answer relevancy).

## Project Structure

```
tcc/
├── main.py                    # Entry point with argparse CLI
├── config/
│   └── default.yaml           # All pipeline configuration
├── src/
│   ├── __init__.py
│   ├── llm.py                 # LLM initialization (Ollama / HuggingFace)
│   ├── document.py            # URL loading and text splitting
│   ├── retriever.py           # Embeddings and Chroma vector store
│   ├── chain.py               # QA chain setup (+ optional few-shot)
│   ├── evaluate.py            # RAGAS evaluation pipeline
│   └── utils.py               # Question processing and model management
├── data/
│   ├── perguntas/             # Question sets per Wikipedia page
│   ├── gabarito/              # Ground-truth answers
│   └── respostas/             # Model outputs organized by topic
│       └── {topic}/ragas/     # RAGAS scores per model
├── reports/
│   ├── performance.csv        # Combined benchmark data
│   ├── plots/                 # Generated charts
│   └── notebooks/             # Analysis notebooks
├── legacy/                    # Previous script versions (for reference)
├── .env.example               # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

## Prerequisites

- **Python** >= 3.12
- **Ollama** installed and running ([ollama.com](https://ollama.com))
- **OpenAI API key** for RAGAS evaluation (stored in `.env`)

## Installation

```bash
git clone https://github.com/MoonHawlk/tcc.git
cd tcc

pip install uv
uv pip install -r requirements.txt

# Copy and fill in your API key
cp .env.example .env
```

## Quick Start

```bash
# Run with default config (zephyr model, GPT-4.5 Wikipedia page)
python main.py

# Override model
python main.py --model "mistral:latest"

# Override model, URL, and question set
python main.py --model "dolphin3:latest" \
               --url "https://en.wikipedia.org/wiki/Brazil" \
               --input data/perguntas/EN_Brazil.csv

# Skip RAGAS evaluation (saves OpenAI tokens)
python main.py --skip-ragas

# Enable few-shot prompting
python main.py --few-shot

# Use HuggingFace backend instead of Ollama
python main.py --backend huggingface

# Use a custom config file
python main.py --config config/experiment_brazil.yaml
```

## Configuration

All parameters are defined in `config/default.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `backend` | - | `"ollama"` | `"ollama"` or `"huggingface"` |
| `ollama` | `model` | `"zephyr:latest"` | Ollama model tag |
| `ollama` | `temperature` | `0.1` | Generation temperature |
| `ollama` | `top_k` | `30` | Token diversity limit |
| `ollama` | `top_p` | `0.8` | Nucleus sampling threshold |
| `document` | `url` | GPT-4.5 Wikipedia | Target page URL |
| `document` | `chunk_size` | `512` | Text splitter chunk size |
| `document` | `chunk_overlap` | `50` | Overlap between chunks |
| `embeddings` | `model_name` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `embeddings` | `retriever_k` | `3` | Top-k documents to retrieve |
| `chain` | `type` | `"refine"` | LangChain chain type |
| `chain` | `few_shot` | `false` | Enable few-shot prompting |
| `ragas` | `judge_model` | `gpt-4.1-nano` | OpenAI model for evaluation |

Any value can be overridden via CLI arguments (see `python main.py --help`).

## Architecture

```
[YAML Config] ──> [argparse CLI overrides]
                           │
                    ┌──────▼──────┐
                    │  build_llm  │  src/llm.py
                    └──────┬──────┘
                    ┌──────▼──────────┐
                    │ load_and_split  │  src/document.py
                    └──────┬──────────┘
                    ┌──────▼────────────┐
                    │ build_retriever   │  src/retriever.py
                    └──────┬────────────┘
                    ┌──────▼──────────────┐
                    │ build_qa_chain      │  src/chain.py
                    └──────┬──────────────┘
                    ┌──────▼──────────────────┐
                    │ process_questions (CSV)  │  src/utils.py
                    └──────┬──────────────────┘
                    ┌──────▼──────────┐
                    │ unload_model    │  src/utils.py
                    └──────┬──────────┘
                    ┌──────▼──────────┐
                    │ run_ragas       │  src/evaluate.py
                    └─────────────────┘
```

## Input & Output

| File | Role |
|------|------|
| `config/default.yaml` | Pipeline configuration |
| `perguntas.csv` | Input questions (column: `pergunta`) |
| `respostas.csv` | Generated answers |
| `scores_ragas.csv` | RAGAS evaluation scores |

## Models Tested

| Model | Ollama Tag |
|-------|-----------|
| Zephyr | `zephyr:latest` |
| Mistral | `mistral:latest` |
| LLaMA 3.1 | `llama3.1:latest` |
| Dolphin 3 | `dolphin3:latest` |

## Notes

- **Memory Management:** After QA processing, the Ollama model is automatically unloaded to free GPU memory before RAGAS evaluation runs.
- **Reproducibility:** Each experiment can be fully defined by a single YAML config file. Copy `config/default.yaml` to create experiment-specific configs.
- **Chunk Tuning:** Adjust `chunk_size` and `chunk_overlap` based on document length for optimal retrieval.
- **Evaluation Cost:** RAGAS evaluation uses the OpenAI API. Use `--skip-ragas` during development to avoid costs.
